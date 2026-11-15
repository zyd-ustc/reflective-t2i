"""GEN LoRA / GEN-full adaptation, edit-progress embedding, anchored state field, and tail generator."""

import copy
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .conditions import edit_prompt
from .runtime import autocast

GEN_ATTN_LINEARS = ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")
GEN_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")
GEN_ATTN_NORMS = ("q_norm_mot_gen", "q_norm_hw_mot_gen", "k_norm_mot_gen", "k_norm_hw_mot_gen")
GEN_LAYER_NORMS = ("input_layernorm_mot_gen", "post_attention_layernorm_mot_gen")


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


def _copy_gen_layer(original):
    """Deep-copy only the GEN weights of one decoder layer, leaving UND & RoPE shared/frozen."""
    layer = _layer_view(original)
    trainable = nn.ModuleDict()
    for name in GEN_ATTN_LINEARS + GEN_ATTN_NORMS:
        if hasattr(layer.self_attn, name):
            copied = copy.deepcopy(getattr(layer.self_attn, name)).requires_grad_(True)
            setattr(layer.self_attn, name, copied)
            trainable[name] = copied
    mlp_copy = copy.deepcopy(original.mlp_mot_gen).requires_grad_(True)
    layer.mlp_mot_gen = mlp_copy
    trainable["mlp_mot_gen"] = mlp_copy
    for name in GEN_LAYER_NORMS:
        if hasattr(layer, name):
            copied = copy.deepcopy(getattr(layer, name)).requires_grad_(True)
            setattr(layer, name, copied)
            trainable[name] = copied
    return layer, trainable


def _merge_lora_into_layers(adapted_layers, weights, *, prefix="adapters."):
    """Fold saved GENLoRA deltas (b @ a) into independent gen_full layer weights."""
    with torch.no_grad():
        for index, layer in enumerate(adapted_layers):
            for group, names in (("self_attn", GEN_ATTN_LINEARS), ("mlp_mot_gen", GEN_MLP_LINEARS)):
                owner = getattr(layer, group)
                for name in names:
                    key_a = f"{prefix}layer_{index}_{name}.a"
                    key_b = f"{prefix}layer_{index}_{name}.b"
                    if key_a in weights and key_b in weights:
                        linear = getattr(owner, name)
                        delta = (
                            weights[key_b].to(device=linear.weight.device, dtype=torch.float32)
                            @ weights[key_a].to(device=linear.weight.device, dtype=torch.float32)
                        )
                        linear.weight.data.add_(delta.to(linear.weight.dtype))


class _GENFlowBase(nn.Module):
    """Shared GEN branch builder and clean-image forward over frozen UND prefixes."""

    def __init__(self, rt, cfg, *, parameterization, lora_rank):
        super().__init__()
        object.__setattr__(self, "rt", rt)
        object.__setattr__(self, "model", rt.model)
        self.config = cfg
        self.parameterization = str(parameterization)
        rt.model.eval().requires_grad_(False)

        if self.parameterization not in ("gen_lora", "gen_full"):
            raise ValueError("Unsupported parameterization: " + self.parameterization)
        if self.parameterization == "gen_lora" and int(lora_rank) < 1:
            raise ValueError("LoRA rank must be positive")

        self.adapters = nn.ModuleDict()
        self.gen_trainable = nn.ModuleDict()
        views, native_views = [], []
        for index, original in enumerate(rt.decoder.layers):
            native_views.append(_layer_view(original))
            if self.parameterization == "gen_lora":
                layer = _layer_view(original)
                for group, names in (("self_attn", GEN_ATTN_LINEARS), ("mlp_mot_gen", GEN_MLP_LINEARS)):
                    owner = getattr(layer, group)
                    for name in names:
                        adapter = GENLoRA(getattr(owner, name), int(lora_rank))
                        self.adapters[f"layer_{index}_{name}"] = adapter
                        setattr(owner, name, adapter)
            else:
                layer, layer_trainable = _copy_gen_layer(original)
                self.gen_trainable[f"layer_{index}"] = layer_trainable
            views.append(layer)

        object.__setattr__(self, "adapted_layers", tuple(views))
        object.__setattr__(self, "native_layers", tuple(native_views))
        if self.parameterization == "gen_full":
            norm_copy = copy.deepcopy(rt.decoder.norm_mot_gen).requires_grad_(True)
            fm_copy = nn.ModuleDict(
                {k: copy.deepcopy(v).requires_grad_(True) for k, v in rt.model.fm_modules.items()}
            )
            self.gen_trainable["norm_mot_gen"] = norm_copy
            self.gen_trainable["fm_modules"] = fm_copy
            object.__setattr__(self, "adapted_norm_mot_gen", norm_copy)
            object.__setattr__(self, "adapted_fm_modules", fm_copy)
        else:
            object.__setattr__(self, "adapted_norm_mot_gen", rt.decoder.norm_mot_gen)
            object.__setattr__(self, "adapted_fm_modules", rt.model.fm_modules)

        self.calls = 0

    def _fm(self, use_adapted):
        return self.adapted_fm_modules if use_adapted else self.model.fm_modules

    def _norm(self, use_adapted):
        return self.adapted_norm_mot_gen if use_adapted else self.rt.decoder.norm_mot_gen

    def gen_inputs(self, z, t, s=None, *, use_adapted=True):
        rt, m = self.rt, self.model
        fm = self._fm(use_adapted)
        batch = z.shape[0]
        gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
        grid = torch.tensor([[gh, gw]] * batch, device=z.device)
        pixels = m.patchify(z, m.patch_size, channel_first=True)
        dtype = fm["vision_model_mot_gen"].embeddings.patch_embedding.weight.dtype
        pixels = pixels.to(dtype)
        flat_pixels = pixels.reshape(batch * gh * gw, -1)
        if fm is m.fm_modules:
            hidden = m.extract_feature(flat_pixels, gen_model=True, grid_hw=grid)
        else:
            hidden = fm["vision_model_mot_gen"](
                pixel_values=flat_pixels, output_hidden_states=False, return_dict=True, grid_hw=grid
            ).last_hidden_state
        hidden = hidden.reshape(batch, rt.tokens, -1)

        expanded = torch.as_tensor(t, device=z.device).reshape(()).expand(batch * rt.tokens)
        condition = fm["timestep_embedder"](expanded).reshape_as(hidden)
        if m.add_noise_scale_embedding:
            sigma = torch.full_like(expanded, rt.noise_scale() / m.noise_scale_max_value)
            condition = condition + fm["noise_scale_embedder"](sigma).reshape_as(hidden)
        if s is not None and hasattr(self, "edit_s_embedder"):
            progress = torch.as_tensor(s, device=z.device, dtype=torch.float32)
            progress = progress.reshape(()).expand(batch * rt.tokens)
            condition = condition + self.edit_s_embedder(progress).reshape_as(hidden).to(hidden.dtype)
        return hidden + condition

    def head(self, hidden, t, *, use_adapted=True):
        rt, m = self.rt, self.model
        fm = self._fm(use_adapted)
        batch = hidden.shape[0]
        if m.use_pixel_head:
            image = hidden.reshape(batch, *rt.grid, -1).permute(0, 3, 1, 2).contiguous()
            return fm["fm_head"](image)
        if m.use_deep_fm_head:
            moment = torch.as_tensor(t, device=hidden.device).reshape(()).expand(batch * rt.tokens)
            clean = fm["fm_head"](
                hidden.reshape(batch * rt.tokens, -1), moment
            ).reshape(batch, rt.tokens, -1)
        else:
            clean = fm["fm_head"](hidden)
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

    def _clean(self, z, t, prefix, *, s=None, use_adapted):
        self.calls += 1
        layers = self.adapted_layers if use_adapted else self.native_layers
        norm = self._norm(use_adapted)
        with autocast(self.rt.device):
            hidden = self.gen_inputs(z, t, s, use_adapted=use_adapted)
            for index in range(len(layers)):
                block = partial(self._block, layers, index, prefix=prefix)
                if self.config["activation_checkpointing"] and torch.is_grad_enabled():
                    hidden = checkpoint(block, hidden, use_reentrant=False)
                else:
                    hidden = block(hidden)
            clean = self.head(norm(hidden), t, use_adapted=use_adapted)
        return clean.float()


class StateEditor(_GENFlowBase):
    """Isochronic state editor: H_theta(q,t,s) minus a frozen anchor H_0(a_t,t).

    Supports both ``gen_lora`` and ``gen_full`` capacity options while always
    preserving the frozen native ``image_clean`` baseline and frozen UND path.
    """

    def __init__(self, rt, cfg):
        super().__init__(
            rt, cfg,
            parameterization=cfg.get("parameterization", "gen_lora"),
            lora_rank=int(cfg["lora_rank"]),
        )
        hidden = int(rt.model.fm_modules["timestep_embedder"].mlp[0].out_features)
        self.edit_s_embedder = type(rt.model.fm_modules["timestep_embedder"])(hidden)
        with torch.no_grad():
            self.edit_s_embedder.mlp[2].weight.zero_()
            self.edit_s_embedder.mlp[2].bias.zero_()
        self.edit_s_embedder.to(rt.device)

        self.baseline_calls = 0
        self.state_calls = 0
        self.prefix_calls = 0
        self.prefix_vision_calls = 0

    @torch.no_grad()
    def prefixes(self, edit_request, *, allow_empty=False):
        """Text-only UND prefills — the edit branch never sees a picture.

        ``image``  unconditional T2I reference for frozen H0;
        ``empty``  canonical unconditional edit prefix (shared with image);
        ``joint``  short instruction prefix, or ``empty`` when dropped.
        """
        instruction = edit_prompt(edit_request, allow_empty=allow_empty)
        image = self._t2i_prefix(None)
        joint = image if instruction is None else self._t2i_prefix(instruction)
        return {
            "image": image,
            "joint": joint,
            "empty": image,
        }

    def _t2i_prefix(self, prompt):
        """One text-only T2I prefill, shaped like ``prefixes`` entries."""
        prefix = self.rt.prefix(prompt)
        self.prefix_calls += 1
        return (prefix.cache, prefix.indexes, None)

    @torch.no_grad()
    def image_clean(self, anchor, t, prefix):
        """Frozen native image-only clean prediction H0(a,t;P)."""
        self.baseline_calls += 1
        return self._clean(anchor, float(t), prefix, s=None, use_adapted=False)

    def state_clean(self, state, t, s, prefix):
        """Joint clean prediction H(q,t,s;P,e)."""
        self.state_calls += 1
        return self._clean(state, float(t), prefix, s=float(s), use_adapted=True)

    def state_velocity(self, state, t, s, joint_prefix, base_prediction):
        """Displacement per unit edit progress: t * [H(q,t,s) - b0]."""
        clean = self.state_clean(state, t, s, joint_prefix)
        return (clean - base_prediction.float()) * float(t)

    def guided_state_velocity(self, state, t, s, joint_prefix, empty_prefix, base_prediction, cfg_scale=1.0):
        """Independent edit CFG: u_empty + w * (u_e - u_empty) around the same H0."""
        conditional = self.state_velocity(state, t, s, joint_prefix, base_prediction)
        scale = float(cfg_scale)
        if scale <= 1.0:
            return conditional
        unconditional = self.state_velocity(state, t, s, empty_prefix, base_prediction)
        return unconditional + scale * (conditional - unconditional)

    def merge_lora_weights_(self, weights):
        """Merge a Stage-1 GEN LoRA checkpoint into this gen_full branch."""
        if self.parameterization != "gen_full":
            raise ValueError("merge_lora_weights_ requires parameterization='gen_full'")
        _merge_lora_into_layers(self.adapted_layers, weights)
        with torch.no_grad():
            for name, param in self.edit_s_embedder.named_parameters():
                key = f"edit_s_embedder.{name}"
                if key not in weights:
                    raise ValueError("Missing edit_s_embedder weight in LoRA checkpoint: " + key)
                param.copy_(weights[key].to(param))


class TailGenerator(_GENFlowBase):
    """Standalone post-edit T2I continuation generator for t in [t_0, 1].

    Uses its own independent GEN adaptation weights on top of the frozen native
    model, with no edit_s_embedder and no source-image conditioning.
    """

    def __init__(self, rt, cfg):
        super().__init__(
            rt, cfg,
            parameterization=cfg.get("tail_parameterization", "gen_lora"),
            lora_rank=int(cfg.get("tail_lora_rank", cfg["lora_rank"])),
        )
        self.tail_calls = 0

    def predict_clean(self, state, t, prefix):
        """Clean image prediction x_1 from z_tau at native time tau."""
        self.tail_calls += 1
        return self._clean(state, float(t), prefix, s=None, use_adapted=True)

    def predict_velocity(self, state, t, prefix):
        """Convert clean prediction to flow velocity (x_pred - z) / max(1 - t, t_eps)."""
        clean = self.predict_clean(state, t, prefix)
        t_eps = float(getattr(self.model.config, "t_eps", self.config["t_eps"]))
        denom = max(1.0 - float(t), t_eps)
        return (clean - state.float()) / denom

    def merge_lora_weights_(self, weights):
        if self.parameterization != "gen_full":
            raise ValueError("merge_lora_weights_ requires tail_parameterization='gen_full'")
        _merge_lora_into_layers(self.adapted_layers, weights)
