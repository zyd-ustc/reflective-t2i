"""GEN LoRA adaptation for the strict isochronic manifold-transition operator."""

import copy
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .conditions import empty_feedback_prompt, feedback_prompt
from .runtime import autocast

GEN_ATTN_LINEARS = ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")
GEN_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")


class GENLoRA(nn.Module):
    """One-sided-random, one-sided-zero adapter on a frozen native linear layer."""

    def __init__(self, base, rank):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        native = self.base(x)
        update = F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))
        return native + update


def module_view(module):
    view = copy.copy(module)
    view._modules = module._modules.copy()
    return view


def _layer_view(original):
    """A shallow copy whose attention and GEN MLP can be wrapped independently."""
    view = module_view(original)
    view.self_attn = module_view(original.self_attn)
    view.mlp_mot_gen = module_view(original.mlp_mot_gen)
    return view


class IsochronicManifoldEditor(nn.Module):
    """Feedback-conditioned short-horizon denoiser for strict isochronic manifold transition.

    Wraps only the GEN attention and MLP projections with zero-initialized LoRA
    while keeping ``fm_head``, ``timestep_embedder``, and UND 100% frozen.
    """

    def __init__(self, rt, cfg):
        super().__init__()
        if str(cfg.get("parameterization", "gen_lora")) != "gen_lora":
            raise ValueError("IsochronicManifoldEditor only supports parameterization='gen_lora'")
        rank = int(cfg["lora_rank"])
        if rank < 1:
            raise ValueError("LoRA rank must be positive")

        object.__setattr__(self, "rt", rt)
        object.__setattr__(self, "model", rt.model)
        self.config = cfg
        self.parameterization = "gen_lora"
        rt.model.eval().requires_grad_(False)

        self.adapters = nn.ModuleDict()
        views, native_views = [], []
        for index, original in enumerate(rt.decoder.layers):
            native_views.append(_layer_view(original))
            layer = _layer_view(original)
            for group, names in (("self_attn", GEN_ATTN_LINEARS), ("mlp_mot_gen", GEN_MLP_LINEARS)):
                owner = getattr(layer, group)
                for name in names:
                    adapter = GENLoRA(getattr(owner, name), rank)
                    self.adapters[f"layer_{index}_{name}"] = adapter
                    setattr(owner, name, adapter)
            views.append(layer)

        object.__setattr__(self, "adapted_layers", tuple(views))
        object.__setattr__(self, "native_layers", tuple(native_views))

        self.calls = 0
        self.clean_calls = 0
        self.native_calls = 0
        self.prefix_calls = 0

    @torch.no_grad()
    def empty_prefix(self):
        """Unconditional T2I prefix shared by anchor extraction, prompt dropout, and edit CFG."""
        prefix = self.rt.prefix(empty_feedback_prompt())
        self.prefix_calls += 1
        return (prefix.cache, prefix.indexes, None)

    @torch.no_grad()
    def feedback_prefix(self, feedback, *, allow_empty=False):
        """Text-only UND prefill using the fixed feedback template."""
        prompt = feedback_prompt(feedback, allow_empty=allow_empty)
        prefix = self.rt.prefix(prompt)
        self.prefix_calls += 1
        return (prefix.cache, prefix.indexes, None)

    def _gen_inputs(self, z, tau):
        rt, m = self.rt, self.model
        fm = m.fm_modules
        batch = z.shape[0]
        gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
        grid = torch.tensor([[gh, gw]] * batch, device=z.device)
        pixels = m.patchify(z, m.patch_size, channel_first=True)
        dtype = fm["vision_model_mot_gen"].embeddings.patch_embedding.weight.dtype
        flat_pixels = pixels.to(dtype).reshape(batch * gh * gw, -1)
        hidden = m.extract_feature(flat_pixels, gen_model=True, grid_hw=grid)
        hidden = hidden.reshape(batch, rt.tokens, -1)

        expanded = torch.as_tensor(tau, device=z.device).reshape(()).expand(batch * rt.tokens)
        condition = fm["timestep_embedder"](expanded).reshape_as(hidden)
        if m.add_noise_scale_embedding:
            sigma = torch.full_like(expanded, rt.noise_scale() / m.noise_scale_max_value)
            condition = condition + fm["noise_scale_embedder"](sigma).reshape_as(hidden)
        return hidden + condition

    def _head(self, hidden, tau):
        rt, m = self.rt, self.model
        fm_head = m.fm_modules["fm_head"]
        batch = hidden.shape[0]
        if m.use_pixel_head:
            image = hidden.reshape(batch, *rt.grid, -1).permute(0, 3, 1, 2).contiguous()
            return fm_head(image.to(next(fm_head.parameters()).dtype))
        if m.use_deep_fm_head:
            moment = torch.as_tensor(tau, device=hidden.device).reshape(()).expand(batch * rt.tokens)
            clean = fm_head(hidden.reshape(batch * rt.tokens, -1), moment).reshape(batch, rt.tokens, -1)
        else:
            clean = fm_head(hidden.to(next(fm_head.parameters()).dtype))
        return m.unpatchify(clean, rt.patch, rt.height, rt.width)

    def _block(self, layers, index, hidden, prefix):
        cache, indexes, mask = prefix
        return layers[index](
            hidden,
            indexes=indexes,
            image_gen_indicators=torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool),
            exist_non_image_gen_tokens=False,
            exist_image_gen_tokens=True,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            update_cache=False,
        )

    def _clean(self, z, tau, prefix, *, use_adapted):
        self.calls += 1
        layers = self.adapted_layers if use_adapted else self.native_layers
        norm = self.rt.decoder.norm_mot_gen
        with autocast(self.rt.device):
            hidden = self._gen_inputs(z, tau)
            for index in range(len(layers)):
                block = partial(self._block, layers, index, prefix=prefix)
                if self.config["activation_checkpointing"] and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, use_reentrant=False)
                else:
                    hidden = block(hidden)
            clean = self._head(norm(hidden), tau)
        return clean.float()

    @torch.no_grad()
    def native_clean(self, state, tau, prefix=None):
        """Frozen native clean prediction H_0(state, tau) used to extract the noise anchor."""
        self.native_calls += 1
        if prefix is None:
            prefix = self.empty_prefix()
        return self._clean(state, float(tau), prefix, use_adapted=False)

    def predict_clean(self, state, tau, prefix):
        """Adapted clean target prediction H_theta(state, tau, feedback)."""
        self.clean_calls += 1
        return self._clean(state, float(tau), prefix, use_adapted=True)

    def predict_guided_clean(self, state, tau, prefix, *, empty_prefix=None, cfg_scale=1.0):
        """Clean target prediction with optional feedback CFG around the unconditional branch."""
        cond = self.predict_clean(state, tau, prefix)
        scale = float(cfg_scale)
        if scale <= 1.0 + 1e-6:
            return cond
        if empty_prefix is None:
            empty_prefix = self.empty_prefix()
        if prefix is empty_prefix:
            raise ValueError("Conditional feedback prefix cannot be identical to empty_prefix under CFG > 1")
        uncond = self.predict_clean(state, tau, empty_prefix)
        return uncond + scale * (cond - uncond)

    def predict_velocity(self, state, tau, prefix, *, empty_prefix=None, cfg_scale=1.0):
        """Convert (guided) clean prediction to flow velocity (H_theta - state) / max(1 - tau, t_eps)."""
        clean = self.predict_guided_clean(
            state, tau, prefix, empty_prefix=empty_prefix, cfg_scale=cfg_scale
        )
        t_eps = float(getattr(self.model.config, "t_eps", self.config["t_eps"]))
        return (clean - state.float()) / max(1.0 - float(tau), t_eps)


def build_editor(rt, cfg):
    """Instantiate the project's sole editor architecture: IsochronicManifoldEditor."""
    return IsochronicManifoldEditor(rt, cfg)
