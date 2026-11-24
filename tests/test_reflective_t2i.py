"""Comprehensive unit tests for the Model-Agnostic Reflective T2I Framework (reflective_t2i_v1)."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn

from qwen_latent_cot.reflective_t2i.core import (
    ConditionBundle,
    IsochronicTransitionV2,
    LatentState,
    LinearFlowGeometry,
    ObserverBackend,
    PipelineConfig,
    ReflectionPolicy,
    ReflectionPolicyConfig,
    ReflectiveT2IPipeline,
    TransitionConfig,
)
from qwen_latent_cot.reflective_t2i.editors import (
    LatentLoRAConfig,
    LatentLoRACorrector,
    TokenMaskContext,
    resolve_layer_indices,
)
from qwen_latent_cot.reflective_t2i.generators.base import GeneratorAdapter
from qwen_latent_cot.reflective_t2i.generators.diffusers.sd35 import SD35Adapter, SD35Config
from qwen_latent_cot.reflective_t2i.training import (
    LossConfig,
    ReflectiveCorrectionLoss,
    RolloutConfig,
    build_training_batch_step,
)
from qwen_latent_cot.reflective_t2i.training.train import TrainConfig, training_step


# ---------------------------------------------------------------------------
# 1. Unified LinearFlowGeometry algebraic invariants (clean_x / Z-Image / SD3.5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pred_type", ["clean_x", "v_to_clean", "v_to_noise"])
@pytest.mark.parametrize("noise_scale", [1.0, 2.0])
def test_linear_flow_geometry_exact_roundtrips(pred_type, noise_scale):
    geom = LinearFlowGeometry(prediction_type=pred_type, noise_scale=noise_scale, t_eps=0.05)
    g = torch.Generator().manual_seed(7)
    x = torch.randn((2, 4, 8, 8), generator=g)
    eps = geom.sample_noise(tuple(x.shape), device=x.device, generator=g)
    p = 0.36
    z = geom.interpolate(x, eps, p)

    if pred_type == "clean_x":
        native_pred = x
    elif pred_type == "v_to_clean":
        native_pred = x - eps
    else:
        native_pred = eps - x

    x_rec = geom.to_clean(z, p, native_pred)
    eps_rec = geom.to_eps(z, p, native_pred)
    v_clean = geom.to_v_clean(z, p, native_pred)
    v_from_clean = geom.from_clean_to_v_clean(z, p, x_rec)

    assert torch.allclose(x_rec, x, atol=1e-5)
    assert torch.allclose(eps_rec, eps, atol=1e-5)
    assert torch.allclose(v_clean, x - eps, atol=1e-5)
    assert torch.allclose(v_from_clean, v_clean, atol=1e-5)

    # Strict isochronic reprojection with eta=0 recovers exact z
    z_reproj = geom.compose_state(x_rec, eps_rec, p, eta=0.0)
    assert torch.allclose(z_reproj, z, atol=1e-5)

    # CPS step with eta=0 matches exact deterministic linear path
    z_next = geom.cps_step(z, p, 0.60, x_rec, eta=0.0)
    assert torch.allclose(z_next, geom.interpolate(x, eps, 0.60), atol=1e-5)


def test_trust_region_clip_enforces_rms_bound():
    ref = torch.ones((2, 4, 8, 8))
    delta = torch.full((2, 4, 8, 8), 5.0)
    clipped = LinearFlowGeometry.trust_region_clip(delta, ref, rho=0.5)
    rms = clipped.square().mean(dim=(1, 2, 3)).sqrt()
    assert torch.allclose(rms, torch.full((2,), 0.5), atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Token-masked single-stream S3-DiT LoRA & dual-stream SD3.5 LoRA isolation
# ---------------------------------------------------------------------------


class _DummySingleStreamBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.to_out[0](self.to_q(x) + self.to_k(x) + self.to_v(x))


class _DummyS3DiT(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 6):
        super().__init__()
        self.noise_refiner = nn.ModuleList([_DummySingleStreamBlock(dim)])
        self.context_refiner = nn.ModuleList([_DummySingleStreamBlock(dim)])
        self.layers = nn.ModuleList([_DummySingleStreamBlock(dim) for _ in range(n_layers)])

    def forward(self, x_img: torch.Tensor, x_txt: torch.Tensor, mask_ctx: TokenMaskContext):
        x_img = self.noise_refiner[0](x_img)
        x_txt = self.context_refiner[0](x_txt)
        unified = torch.cat([x_img, x_txt], dim=1)
        bsz, total_len, _ = unified.shape
        img_len = x_img.shape[1]
        mask = torch.zeros((bsz, total_len, 1), dtype=torch.bool, device=unified.device)
        mask[:, :img_len, 0] = True
        with mask_ctx.apply(mask):
            for layer in self.layers:
                unified = layer(unified)
        return unified[:, :img_len], unified[:, img_len:]


def test_single_stream_token_masked_lora_zero_init_and_text_isolation():
    torch.manual_seed(0)
    model = _DummyS3DiT(dim=16, n_layers=6)
    x_img = torch.randn(2, 5, 16)
    x_txt = torch.randn(2, 4, 16)
    mask_ctx = TokenMaskContext()

    with torch.no_grad():
        base_img, base_txt = model(x_img, x_txt, mask_ctx)

    cfg = LatentLoRAConfig(
        rank=4,
        layer_scope="middle_third",
        block_prefix="layers",
        include_prefixes=("layers.",),
        exclude_substrings=("context_refiner",),
        target_suffixes=("to_q", "to_k", "to_v", "to_out.0"),
        masked_prefixes=("layers.",),
    )
    corrector = LatentLoRACorrector(model, cfg, num_blocks=6, token_mask_ctx=mask_ctx)

    # Only layers 2 and 3 (middle third of 6) should be wrapped; context_refiner untouched
    assert resolve_layer_indices(6, "middle_third") == {2, 3}
    assert all(name.startswith(("layers.2.", "layers.3.")) for name in corrector.wrapped)
    assert not any("context_refiner" in name for name in corrector.wrapped)

    # Zero-init parity: active corrector == frozen base bit-for-bit
    corrector.set_active(True)
    with torch.no_grad():
        z_img, z_txt = model(x_img, x_txt, mask_ctx)
    assert torch.equal(z_img, base_img)
    assert torch.equal(z_txt, base_txt)

    # Non-zero LoRA weights: image tokens change, text tokens stay EXACTLY identical!
    with torch.no_grad():
        for name, p in corrector.trainable_parameters().items():
            if ".lora_b." in name:
                p.fill_(0.25)

    out_img, out_txt = model(x_img, x_txt, mask_ctx)
    assert not torch.allclose(out_img, base_img)
    assert torch.equal(out_txt, base_txt)

    # Text-only loss must produce ZERO gradient on LoRA parameters!
    out_txt.square().mean().backward()
    for p in corrector.trainable_parameters().values():
        assert p.grad is None or float(p.grad.abs().max().item()) == 0.0

    # Round-trip trainable checkpoint
    sd = corrector.state_dict_trainable()
    with torch.no_grad():
        for p in corrector.trainable_parameters().values():
            p.zero_()
    corrector.load_state_dict_trainable(sd, strict=True)
    out_img_reloaded, _ = model(x_img, x_txt, mask_ctx)
    assert torch.allclose(out_img_reloaded, out_img, atol=1e-6)


class _DummySD3Block(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.attn = SimpleNamespace(
            to_q=nn.Linear(dim, dim),
            to_k=nn.Linear(dim, dim),
            to_v=nn.Linear(dim, dim),
            to_out=nn.ModuleList([nn.Linear(dim, dim)]),
            add_q_proj=nn.Linear(dim, dim),
            add_k_proj=nn.Linear(dim, dim),
            add_v_proj=nn.Linear(dim, dim),
            to_add_out=nn.Linear(dim, dim),
        )
        # Register as proper nn.Module submodules
        attn_mod = nn.Module()
        for k, v in self.attn.__dict__.items():
            setattr(attn_mod, k, v)
        self.attn = attn_mod
        ff_ctx = nn.Module()
        ff_ctx.net = nn.ModuleList([nn.Linear(dim, dim)])
        self.ff_context = ff_ctx


class _DummySD3Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.transformer_blocks = nn.ModuleList([_DummySD3Block(8) for _ in range(6)])

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict=False):
        return (torch.zeros_like(hidden_states),)


def test_sd35_adapter_wraps_only_image_stream_and_splits_clip_t5_conditions():
    recorded = {}

    class _DummyPipe:
        vae_scale_factor = 8

        def __init__(self):
            self.transformer = _DummySD3Transformer()
            self.vae = nn.Conv2d(3, 4, 1)
            self.vae.config = SimpleNamespace(shift_factor=0.0, scaling_factor=1.0)
            self.scheduler = SimpleNamespace(
                config=SimpleNamespace(num_train_timesteps=1000),
                sigmas=torch.linspace(1.0, 0.0, 5),
                set_timesteps=lambda n, device=None: None,
            )

        def encode_prompt(self, prompt, prompt_2, prompt_3, **kwargs):
            recorded["clip"] = prompt
            recorded["t5"] = prompt_3
            return torch.zeros(1, 4, 8), None, torch.zeros(1, 8), None

    adapter = SD35Adapter(_DummyPipe(), SD35Config(width=64, height=64, lora_rank=4, layer_scope="middle_third"))
    assert adapter.corrector is not None
    wrapped_names = list(adapter.corrector.wrapped.keys())
    assert len(wrapped_names) == 8  # 2 middle blocks * 4 image attn projections
    assert all("add_" not in n and "ff_context" not in n for n in wrapped_names)

    adapter.encode_edit_condition(
        feedback="Remove 1 hotdog so that it contains 5 instead of 6",
        original_prompt="Five hotdogs on a plate",
        target_rewrite="A photo of five hotdogs on a plate",
        condition_mode="native_prompt_plus_feedback",
    )
    assert recorded["clip"] == "A photo of five hotdogs on a plate"
    assert "Original: Five hotdogs on a plate" in recorded["t5"]
    assert "Remove 1 hotdog" in recorded["t5"]


# ---------------------------------------------------------------------------
# 3. Zero-bbox Delta-Weighted + Cosine Residual Loss
# ---------------------------------------------------------------------------


def test_delta_weighted_and_cosine_loss_safety_and_small_object_focus():
    source = torch.zeros((2, 4, 16, 16))
    target = source.clone()
    # Small 4x4 added object (only 6.25% of canvas)
    target[:, :, 2:6, 2:6] = 2.0

    loss_fn = ReflectiveCorrectionLoss(LossConfig(delta_weight_lambda=4.0, delta_weight_max=8.0, cosine_lambda=0.1))

    # Case A: Step-1 zero-init (pred == source -> pred_delta == 0) must stay finite with clean grads
    pred_zero = source.clone().requires_grad_(True)
    loss_zero, stats_zero = loss_fn(pred_zero, target, source)
    assert torch.isfinite(loss_zero)
    loss_zero.backward()
    assert torch.isfinite(pred_zero.grad).all()
    assert stats_zero["cos_active_frac"] == 0.0

    # Case B: Small-object region receives strictly higher gradient magnitude than background
    pred = (source + 0.1 * torch.randn_like(source)).detach().requires_grad_(True)
    loss, stats = loss_fn(pred, target, source)
    loss.backward()
    obj_grad = pred.grad[:, :, 2:6, 2:6].abs().mean().item()
    bg_grad = pred.grad[:, :, 8:16, 8:16].abs().mean().item()
    assert obj_grad > 5.0 * bg_grad
    assert stats["edit_weight_ratio"] > 1.0

    # Case C: Identity-dropout sample (target == source) stays finite
    pred_id = (source + 0.05).detach().requires_grad_(True)
    loss_id, _ = loss_fn(pred_id, source, source)
    assert torch.isfinite(loss_id)
    loss_id.backward()
    assert torch.isfinite(pred_id.grad).all()


# ---------------------------------------------------------------------------
# 4. ReflectionPolicy: JSON parsing, confidence gate, bidirectional add/remove recheck
# ---------------------------------------------------------------------------


class _ScriptedObserver(ObserverBackend):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def query_image(self, image, prompt, *, system_prompt=None, max_new_tokens=384, temperature=0.0) -> str:
        resp = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return resp


def test_reflection_policy_bidirectional_add_and_remove_gates():
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    policy = ReflectionPolicy(ReflectionPolicyConfig(confidence_threshold=0.75))

    # 1. False `remove` intercepted by focused_remove_recheck (preview actually has 5, not 6!)
    obs_false_rm = _ScriptedObserver(
        [
            json.dumps(
                {
                    "observed_objects": [{"name": "hotdog", "count": 6}],
                    "missing_or_wrong": ["too many hotdogs"],
                    "match": False,
                    "action": "remove",
                    "instruction": "Remove 1 hotdog so that it contains exactly 5 hotdogs instead of the current 6, filling the vacated areas naturally; keep everything else unchanged.",
                    "target_rewrite": "Five hotdogs on a tray",
                    "coarse_region_hint": "right side",
                    "confidence": 0.90,
                }
            ),
            json.dumps({"instances": ["1", "2", "3", "4", "5"], "count": 5, "uncertain": False}),
        ]
    )
    dec = policy.reflect(obs_false_rm, img, "a photo of five hotdogs")
    assert dec.outcome == "gated_noop"
    assert dec.gated_by == "focused_remove_recheck"
    assert obs_false_rm.calls == 2

    # 2. True `add` passes focused_add_recheck when recheck confirms missing count
    obs_true_add = _ScriptedObserver(
        [
            "<think>checking</think>\n```json\n"
            + json.dumps(
                {
                    "observed_objects": [{"name": "bee", "count": 1}],
                    "missing_or_wrong": ["missing 1 bee"],
                    "match": False,
                    "action": "add",
                    "instruction": "Add 1 bee so that it contains exactly 2 bees instead of the current 1, placing the new ones in the open grass on the right; keep everything else unchanged.",
                    "target_rewrite": "Two bees on grass",
                    "coarse_region_hint": "open grass on the right",
                    "confidence": 0.88,
                }
            )
            + "\n```",
            json.dumps({"instances": ["left bee"], "count": 1, "uncertain": False}),
        ]
    )
    dec2 = policy.reflect(obs_true_add, img, "a photo of two bees")
    assert dec2.outcome == "edit"
    assert dec2.action == "add"
    assert dec2.coarse_region_hint == "open grass on the right"

    # 3. Low confidence triggers gated_noop without spending a 2nd VLM call
    obs_low_conf = _ScriptedObserver(
        [
            json.dumps(
                {
                    "match": False,
                    "action": "attr",
                    "instruction": "Change the red cup to blue.",
                    "confidence": 0.40,
                }
            )
        ]
    )
    dec3 = policy.reflect(obs_low_conf, img, "a photo of a blue cup")
    assert dec3.outcome == "gated_noop"
    assert dec3.gated_by == "low_confidence"
    assert obs_low_conf.calls == 1


# ---------------------------------------------------------------------------
# 5. End-to-end Pipeline (all 5 causal arms A-E) + SOAR training_step
# ---------------------------------------------------------------------------


class _StubFlowGenerator(GeneratorAdapter):
    def __init__(self):
        self._geom = LinearFlowGeometry(prediction_type="v_to_clean", noise_scale=1.0, t_eps=0.05)
        self._device = torch.device("cpu")
        self.backbone = nn.Sequential(nn.Linear(16, 16))
        self.layers = nn.ModuleList([_DummySingleStreamBlock(16)])
        self.corrector = LatentLoRACorrector(
            self.layers,
            LatentLoRAConfig(
                rank=4,
                layer_scope="all",
                block_prefix="0",
                include_prefixes=("0.",),
                target_suffixes=("to_q", "to_k", "to_v", "to_out.0"),
            ),
            num_blocks=1,
        )

    @property
    def geometry(self) -> LinearFlowGeometry:
        return self._geom

    @property
    def device(self) -> torch.device:
        return self._device

    def progress_grid(self) -> list[float]:
        return [i / 10.0 for i in range(11)]

    def sample_initial_noise(self, *, seed: int, batch_size: int = 1) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return self._geom.sample_noise((batch_size, 4, 2, 2), device=self._device, generator=g)

    def encode_prompt(self, text: str | None) -> ConditionBundle:
        bias = 0.0 if text is None else (sum(ord(c) for c in text) % 17) / 17.0
        return ConditionBundle(raw_text=text, payload=torch.tensor(bias, dtype=torch.float32))

    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor:
        return rgb_minus1_1[:, :4, :2, :2] if rgb_minus1_1.shape[1] >= 4 else rgb_minus1_1.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)[:, :, :2, :2]

    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image:
        return Image.new("RGB", (16, 16), (120, 120, 120))

    def predict_clean(self, z, p, cond, *, use_corrector=False) -> torch.Tensor:
        self.corrector.set_active(bool(use_corrector))
        bsz = z.shape[0]
        flat = z.float().reshape(bsz, 1, 16) + float(cond.payload) * 0.1
        out = self.layers[0](flat).reshape_as(z)
        self.corrector.set_active(False)
        return self._geom.to_clean(z, p, out)


def test_end_to_end_pipeline_all_five_arms_and_soar_training_step():
    gen = _StubFlowGenerator()
    # Give LoRA non-zero B so correction actually alters the trajectory
    with torch.no_grad():
        for name, p in gen.corrector.trainable_parameters().items():
            if ".lora_b." in name:
                p.fill_(0.1)

    obs = _ScriptedObserver(
        [
            json.dumps(
                {
                    "observed_objects": [{"name": "apple", "count": 1}],
                    "match": False,
                    "action": "add",
                    "instruction": "Add 1 red apple so that it contains exactly 2 red apples instead of the current 1; keep everything else unchanged.",
                    "target_rewrite": "Two red apples on a table",
                    "coarse_region_hint": "on the right",
                    "confidence": 0.95,
                }
            ),
            json.dumps({"instances": ["left apple"], "count": 1, "uncertain": False}),
        ]
    )
    pipe = ReflectiveT2IPipeline(
        gen,
        obs,
        PipelineConfig(
            read_progress=0.36,
            stop_progress=0.60,
            lookahead_steps=3,
            transition=TransitionConfig(
                read_progress=0.36,
                stop_progress=0.60,
                anchor_mode="conditional_native",
                edit_cfg_scale=1.4,
                trust_region_rho=0.65,
            ),
        ),
    )

    res_a = pipe.run("a photo of two red apples", seed=42, arm="A_native")
    assert not res_a.telemetry["triggered"]

    obs_b = _ScriptedObserver(list(obs.responses))
    pipe.observer = obs_b
    res_b = pipe.run("a photo of two red apples", seed=42, arm="B_rewrite_resample")
    assert res_b.telemetry["triggered"]

    obs_c = _ScriptedObserver(list(obs.responses))
    pipe.observer = obs_c
    res_c = pipe.run("a photo of two red apples", seed=42, arm="C_reflective")
    assert res_c.telemetry["triggered"]
    assert res_c.transitions[0].telemetry["anchor_mode"] == "conditional_native"
    assert res_c.transitions[0].telemetry["anchor_nfe"] == 0  # Reused cached prefix clean estimate!

    res_d = pipe.run("a photo of two red apples", seed=42, arm="D_oracle_feedback", oracle_instruction="Add 1 red apple.")
    res_e = pipe.run("a photo of two red apples", seed=42, arm="E_shuffled_feedback", shuffled_instruction="Change sky to green.")
    assert not torch.allclose(res_d.final_tensor, res_e.final_tensor)

    # Verify SOAR self-rollout + S-anchored dropout + backward pass updates ONLY LoRA params
    base_snapshot = {
        n: p.detach().clone()
        for n, p in gen.layers.named_parameters()
        if "lora_" not in n
    }
    sample = {
        "uid": "demo_1",
        "source_rgb": torch.zeros(3, 4, 4),
        "target_rgb": torch.ones(3, 4, 4),
        "valid": torch.ones(1, 4, 4),
        "edit_instruction": "Add 1 red apple.",
        "original_prompt": "One red apple.",
        "target_caption": "Two red apples.",
        "semantic_task": "count",
    }
    loss_fn = ReflectiveCorrectionLoss(LossConfig())
    # Force SOAR rollout branch
    t_cfg_soar = TrainConfig(
        rollout=RolloutConfig(soar_rollout_prob=1.0, edit_prompt_dropout=0.0),
    )
    loss_soar, metrics_soar = training_step(gen, sample, loss_fn, t_cfg_soar, rng=random.Random(1))
    assert metrics_soar["soar_used"] == 1
    assert torch.isfinite(loss_soar)
    loss_soar.backward()

    # Force S-anchored identity dropout branch
    t_cfg_drop = TrainConfig(
        rollout=RolloutConfig(edit_prompt_dropout=1.0, dropout_target="source_identity"),
    )
    _, metrics_drop = training_step(gen, sample, loss_fn, t_cfg_drop, rng=random.Random(2))
    assert metrics_drop["branch"] == "dropout_identity"
    assert metrics_drop["prompt_dropped"] == 1

    # Frozen base weights must remain untouched and have zero grad
    for n, p in gen.layers.named_parameters():
        if "lora_" not in n:
            assert p.grad is None
            assert torch.equal(p.detach(), base_snapshot[n])


def test_compile_concise_imperative_strips_endpoints_and_compound_clauses():
    from qwen_latent_cot.reflective_t2i.core import compile_concise_imperative

    compound_add = (
        "Add 1 backpack so that it contains exactly 7 backpacks instead of the current 6, "
        "placing the new one at the bottom of the left column; keep everything else unchanged. "
        "Add 2 rabbits so that it contains exactly 4 rabbits instead of the current 2."
    )
    assert compile_concise_imperative(compound_add, action="add") == "Add 1 backpack at the bottom of the left column."

    verbose_remove = (
        "Remove 1 red chair so that it contains exactly 5 red chairs instead of the current 6, "
        "filling the vacated areas naturally; keep everything else unchanged."
    )
    assert compile_concise_imperative(verbose_remove, action="remove") == "Remove 1 red chair."

    verbose_attr = "Add a sparkling effect to the pig; keep everything else unchanged."
    assert compile_concise_imperative(verbose_attr, action="attr") == "Add a sparkling effect to the pig."

    add_with_hint = (
        "Add one sparkling monkey so that it contains exactly three monkeys instead of the current two; "
        "keep everything else unchanged."
    )
    assert (
        compile_concise_imperative(add_with_hint, action="add", coarse_region_hint="open space on the left")
        == "Add 1 sparkling monkey in the open space on the left."
    )
