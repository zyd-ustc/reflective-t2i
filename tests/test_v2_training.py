"""Unit tests for v2 online rollout, remaining-endpoint supervision, prompt dropout/CFG, GEN-full, and standalone tail generator."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn

from qwen_latent_cot.u15_target_correction.checkpoint import (
    canonical_parameters,
    editor_identity,
    load_init_weights,
    load_stage_weights,
    save,
)
from qwen_latent_cot.u15_target_correction.conditions import edit_prompt, empty_edit_prompt
from qwen_latent_cot.u15_target_correction.config import load_config, validate_config
from qwen_latent_cot.u15_target_correction.data import (
    LinearManifest,
    TailManifest,
    TailSampler,
    TrainingSampler,
    classify_semantic_task,
)
from qwen_latent_cot.u15_target_correction.field import TailT2I, edit_state, phase_grid, rollout_prefix
from qwen_latent_cot.u15_target_correction import inference as inference_mod
from qwen_latent_cot.u15_target_correction.inference import edit_pair, t2i_feedback
from qwen_latent_cot.u15_target_correction.loss import endpoint_flow_matching, state_flow_matching, tail_flow_matching
from qwen_latent_cot.u15_target_correction.model import GENLoRA, StateEditor, TailGenerator
from qwen_latent_cot.u15_target_correction.train import _tail_update, _update


CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "u15_isochronic_state"


class _DummyNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight


class _DummyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj_mot_gen = nn.Linear(dim, dim, bias=False)
        self.k_proj_mot_gen = nn.Linear(dim, dim, bias=False)
        self.v_proj_mot_gen = nn.Linear(dim, dim, bias=False)
        self.o_proj_mot_gen = nn.Linear(dim, dim, bias=False)
        self.q_norm_mot_gen = _DummyNorm(dim)
        self.k_norm_mot_gen = _DummyNorm(dim)

    def forward(self, x, prefix_bias=0.0):
        q = self.q_norm_mot_gen(self.q_proj_mot_gen(x))
        k = self.k_norm_mot_gen(self.k_proj_mot_gen(x))
        v = self.v_proj_mot_gen(x)
        return self.o_proj_mot_gen(v + 0.05 * (q + k) + prefix_bias)


class _DummyMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, dim, bias=False)
        self.down_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _DummyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attn = _DummyAttention(dim)
        self.mlp_mot_gen = _DummyMLP(dim)
        self.input_layernorm_mot_gen = _DummyNorm(dim)
        self.post_attention_layernorm_mot_gen = _DummyNorm(dim)

    def forward(
        self,
        hidden,
        indexes=None,
        image_gen_indicators=None,
        exist_non_image_gen_tokens=False,
        exist_image_gen_tokens=True,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        update_cache=False,
    ):
        del indexes, image_gen_indicators, exist_non_image_gen_tokens, exist_image_gen_tokens, attention_mask, use_cache, update_cache
        bias = float(past_key_values) if isinstance(past_key_values, (int, float)) else 0.0
        h = hidden + self.self_attn(self.input_layernorm_mot_gen(hidden), prefix_bias=bias)
        return h + self.mlp_mot_gen(self.post_attention_layernorm_mot_gen(h))


class _DummyVisionGen(nn.Module):
    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.embeddings = SimpleNamespace(patch_embedding=nn.Linear(patch_dim, hidden_dim, bias=False))

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True, grid_hw=None):
        del output_hidden_states, return_dict, grid_hw
        return SimpleNamespace(last_hidden_state=self.embeddings.patch_embedding(pixel_values))


class _DummyScalarEmbed(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x.float().reshape(-1, 1))


class _DummyFMHead(nn.Module):
    def __init__(self, hidden_dim, patch_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, patch_dim, bias=False)

    def forward(self, hidden):
        return self.proj(hidden)


class _DummyNEO(nn.Module):
    def __init__(self, hidden_dim=8, patch_size=4, layers=2):
        super().__init__()
        patch_dim = 3 * patch_size * patch_size
        self.patch_size = patch_size
        self.use_pixel_head = False
        self.use_deep_fm_head = False
        self.add_noise_scale_embedding = True
        self.noise_scale_max_value = 1.0
        self.config = SimpleNamespace(hidden_size=hidden_dim, t_eps=0.05)
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(
                layers=nn.ModuleList([_DummyLayer(hidden_dim) for _ in range(layers)]),
                norm_mot_gen=_DummyNorm(hidden_dim),
            )
        )
        self.fm_modules = nn.ModuleDict(
            dict(
                vision_model_mot_gen=_DummyVisionGen(patch_dim, hidden_dim),
                timestep_embedder=_DummyScalarEmbed(hidden_dim),
                noise_scale_embedder=_DummyScalarEmbed(hidden_dim),
                fm_head=_DummyFMHead(hidden_dim, patch_dim),
            )
        )

    def patchify(self, x, patch_size, channel_first=True):
        del channel_first
        b, c, h, w = x.shape
        gh, gw = h // patch_size, w // patch_size
        x = x.reshape(b, c, gh, patch_size, gw, patch_size)
        return x.permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * patch_size * patch_size)

    def unpatchify(self, clean, patch_size, height, width):
        b = clean.shape[0]
        gh, gw = height // patch_size, width // patch_size
        x = clean.reshape(b, gh, gw, 3, patch_size, patch_size)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(b, 3, height, width)

    def extract_feature(self, flat_pixels, gen_model=True, grid_hw=None):
        del gen_model, grid_hw
        return self.fm_modules["vision_model_mot_gen"].embeddings.patch_embedding(flat_pixels)


class _DummyRuntime:
    def __init__(self, width=16, height=16, patch=4, hidden_dim=8):
        self.device = torch.device("cpu")
        self.width = width
        self.height = height
        self.patch = patch
        self.grid = (height // patch, width // patch)
        self.tokens = self.grid[0] * self.grid[1]
        self.model = _DummyNEO(hidden_dim=hidden_dim, patch_size=patch, layers=2)
        self.decoder = self.model.language_model.model
        self.identity = dict(
            checkpoint="mock", source_commit="mock", width=width, height=height, patch=patch,
            tokens=self.tokens, noise_scale=1.0, time_length=5, hidden_size=hidden_dim,
        )

    def noise_scale(self):
        return 1.0

    def times(self):
        return [0.0, 0.25, 0.36, 0.5, 1.0]

    def prefix(self, prompt):
        assert prompt is None or isinstance(prompt, str), f"expected str|None prompt, got {type(prompt)}"
        code = 0.0 if prompt is None else float((sum(ord(ch) for ch in prompt) % 7) + 1) * 0.1
        return SimpleNamespace(cache=code, indexes=torch.zeros(self.tokens))

    def edit_prefix(self, image_prompt, edit_prompt_text):
        img = self.prefix(image_prompt)
        joint_code = img.cache + (0.0 if not edit_prompt_text else float((sum(ord(ch) for ch in edit_prompt_text) % 5) + 1) * 0.2)
        return img, SimpleNamespace(cache=joint_code, indexes=torch.zeros(self.tokens))

    def guided(self, positive, negative, cfg_scale):
        return negative + float(cfg_scale) * (positive - negative)

    def predict(self, state, time, prefix):
        del prefix
        return torch.zeros_like(state)

    def cfg_velocity(self, state, time, cond, uncond):
        return torch.zeros_like(state)

    def review(self, image, prompt, max_new_tokens=64):
        del image, prompt, max_new_tokens
        return "yes"


def _make_v2_cfg(component="editor", parameterization="gen_lora", **overrides):
    cfg = {
        "kind": "u15_isochronic_state_flow",
        "protocol_version": 1,
        "stage": 2,
        "component": component,
        "parameterization": parameterization,
        "tail_parameterization": parameterization if component == "tail" else "gen_lora",
        "tail_lora_rank": 4,
        "condition_protocol": "pure_text_und_v1",
        "width": 16,
        "height": 16,
        "num_steps": 4,
        "edit_steps": 4,
        "lora_rank": 4,
        "t_eps": 0.05,
        "cfg_scale": 4.0,
        "image_cfg_scale": 1.0,
        "cfg_norm": "none",
        "cfg_interval": [0.0, 1.0],
        "state_times": [0.36],
        "state_time_probs": [1.0],
        "read_time": 0.36,
        "lookahead_steps": 2,
        "timestep_shift": 3.0,
        "lr": 1e-4,
        "lr_final": 1e-5,
        "warmup_steps": 10,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "global_batch": 1,
        "save_every": 10,
        "max_steps": 100,
        "updates_per_pair": 1,
        "activation_checkpointing": False,
        "keep_instruction": "keep unchanged",
        "keep_fraction": 0.0,
        "role_weights": {"L_edit": 1.0} if component == "editor" else {"tail_pair": 1.0},
        "source_weights": {"linear_pair": 1.0} if component == "editor" else {"tail_pair": 1.0},
        "edit_prompt_dropout": 0.1,
        "edit_cfg_scale": 1.5,
        "rollout_prob": 0.6,
        "terminal_rollout_prob": 0.1,
        "rollout_cfg_prob": 0.2,
        "rollout_cfg_scale": 1.5,
        "tail_t_min": 0.36,
        "tail_t_max": 1.0,
        "tail_native_prob": 0.2,
        "tail_prompt_dropout": 0.1,
        "allow_derived_tail": False,
        "seed": 17,
    }
    cfg.update(overrides)
    return cfg


def test_conditions_empty_prompt():
    with pytest.raises(ValueError):
        edit_prompt("   ")
    assert empty_edit_prompt() is None
    assert edit_prompt("", allow_empty=True) is None
    assert edit_prompt("add one red apple") == "Edit request: add one red apple"


def test_endpoint_loss_algebraic_equivalence():
    torch.manual_seed(0)
    a_t = torch.randn(1, 3, 16, 16)
    b_t = torch.randn(1, 3, 16, 16)
    valid = torch.ones(1, 1, 16, 16)
    t = 0.36
    s = 0.25
    q_ideal = (1.0 - s) * a_t + s * b_t
    v_pred = (b_t - a_t) * 0.7 + 0.1 * torch.randn_like(a_t)
    b_hat = q_ideal + (1.0 - s) * v_pred

    # On the ideal path, ||(b_hat - b_t)/t||^2 == (1 - s)^2 * ||(v_pred - (b_t - a_t))/t||^2.
    loss_ep = endpoint_flow_matching(b_hat, b_t, t, valid)
    loss_ifm = state_flow_matching(v_pred, b_t - a_t, t, valid)
    assert torch.allclose(loss_ep, ((1.0 - s) ** 2) * loss_ifm, atol=1e-6, rtol=1e-5)
    assert endpoint_flow_matching(b_t, b_t, t, valid).item() == pytest.approx(0.0, abs=1e-7)


def test_tail_flow_matching_minimizer():
    torch.manual_seed(1)
    target = torch.randn(1, 3, 16, 16)
    state = torch.randn(1, 3, 16, 16)
    valid = torch.ones(1, 1, 16, 16)
    assert tail_flow_matching(target, state, target, 0.36, 0.05, valid).item() == pytest.approx(0.0, abs=1e-7)
    # Near tau=0.99, clamp at t_eps=0.05 prevents division by zero and preserves x_b minimizer.
    assert tail_flow_matching(target, state, target, 0.99, 0.05, valid).item() == pytest.approx(0.0, abs=1e-7)
    wrong = tail_flow_matching(target + 0.5, state, target, 0.99, 0.05, valid)
    assert wrong.item() > 0.0


def test_gen_full_isolation_and_lora_merge_equivalence():
    torch.manual_seed(2)
    rt = _DummyRuntime()
    cfg_lora = _make_v2_cfg(parameterization="gen_lora")
    cfg_full = _make_v2_cfg(parameterization="gen_full")

    lora_editor = StateEditor(rt, cfg_lora).train()
    # Non-zero LoRA update
    for name, module in lora_editor.adapters.items():
        if isinstance(module, GENLoRA):
            nn.init.normal_(module.a, std=0.05)
            nn.init.normal_(module.b, std=0.05)
    nn.init.normal_(lora_editor.edit_s_embedder.mlp[2].weight, std=0.1)

    state = torch.randn(1, 3, 16, 16)
    epref = lora_editor.prefixes("add two cups")
    base_lora = lora_editor.image_clean(state, 0.36, epref["image"])
    vel_lora = lora_editor.state_velocity(state, 0.36, 0.5, epref["joint"], base_lora)

    full_editor = StateEditor(rt, cfg_full).train()
    full_editor.merge_lora_weights_(canonical_parameters(lora_editor))
    base_full = full_editor.image_clean(state, 0.36, epref["image"])
    vel_full = full_editor.state_velocity(state, 0.36, 0.5, epref["joint"], base_full)

    assert torch.allclose(base_lora, base_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(vel_lora, vel_full, atol=1e-5, rtol=1e-5)

    # Gradient step on gen_full must NEVER change rt.model or frozen base prediction.
    opt = torch.optim.SGD(canonical_parameters(full_editor).values(), lr=0.1)
    loss = full_editor.state_velocity(state, 0.36, 0.25, epref["joint"], base_full).pow(2).mean()
    loss.backward()
    opt.step()
    base_after = full_editor.image_clean(state, 0.36, epref["image"])
    assert torch.allclose(base_full, base_after, atol=1e-6)


def test_editor_and_tail_coexistence_and_checkpoint_init(tmp_path):
    torch.manual_seed(3)
    rt = _DummyRuntime()
    cfg_lora = _make_v2_cfg(parameterization="gen_lora")
    cfg_full = _make_v2_cfg(parameterization="gen_full")
    cfg_tail = _make_v2_cfg(component="tail", parameterization="gen_lora")

    editor = StateEditor(rt, cfg_lora).train()
    for m in editor.adapters.values():
        nn.init.normal_(m.a, std=0.05)
        nn.init.normal_(m.b, std=0.05)
    tail = TailGenerator(rt, cfg_tail).train()
    for m in tail.adapters.values():
        nn.init.normal_(m.a, std=0.05)
        nn.init.normal_(m.b, std=0.05)

    # Mutating tail must not alter editor outputs, and vice versa.
    z = torch.randn(1, 3, 16, 16)
    epref = editor.prefixes("remove one dog")
    base = editor.image_clean(z, 0.36, epref["image"])
    tpref = rt.prefix("one dog")
    v_before = editor.state_velocity(z, 0.36, 0.0, epref["joint"], base)
    loss_t = tail.predict_clean(z, 0.5, (tpref.cache, tpref.indexes, None)).pow(2).mean()
    loss_t.backward()
    for p in tail.parameters():
        if p.grad is not None:
            p.data.add_(-0.1 * p.grad)
    v_after = editor.state_velocity(z, 0.36, 0.0, epref["joint"], base)
    assert torch.allclose(v_before, v_after, atol=1e-6)

    # Save LoRA checkpoint and init-load into gen_full editor (Stage 1 -> Stage 2 upgrade).
    ckpt = tmp_path / "stage1_lora"
    opt = torch.optim.AdamW(canonical_parameters(editor).values(), lr=1e-4)
    dummy_sampler = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda s: None)
    save(
        ckpt, editor, opt, dummy_sampler, 10, cfg_lora,
        editor_identity(rt, cfg_lora, component="editor"), rt.device,
        data_version="v", input_domains=["t"], total_steps=10, budget=10,
    )
    upgraded = StateEditor(rt, cfg_full).eval()
    load_init_weights(ckpt, upgraded, editor_identity(rt, cfg_full, component="editor"))
    base_up = upgraded.image_clean(z, 0.36, epref["image"])
    assert torch.allclose(
        editor.state_velocity(z, 0.36, 0.5, epref["joint"], base),
        upgraded.state_velocity(z, 0.36, 0.5, epref["joint"], base_up),
        atol=1e-5, rtol=1e-5,
    )


def test_update_rollout_dropout_cfg_and_tail_update():
    torch.manual_seed(4)
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(parameterization="gen_lora")
    grid = phase_grid(rt, cfg)
    editor = StateEditor(rt, cfg).train()
    valid = torch.ones(1, 1, 16, 16)
    sample_base = dict(
        uid="u1", role="L_edit", source_kind="linear_pair",
        source=torch.randn(1, 3, 16, 16),
        target=torch.randn(1, 3, 16, 16),
        valid=valid,
        edit_request="add one apple",
        image_prompt="",
        keep=False,
        time=0.36,
    )

    # 1. Forced prompt dropout (keep=False still targets x_b, empty condition)
    cfg_drop = dict(cfg, edit_prompt_dropout=0.999999, rollout_prob=0.0, terminal_rollout_prob=0.0)
    editor.zero_grad(set_to_none=True)
    loss, fields = _update(editor, sample_base, cfg_drop, grid, random.Random(0))
    assert fields["mode"] == "ifm"
    assert fields["prompt_dropped"] == 1
    assert fields["keep"] == 0
    loss.backward()
    assert any(p.grad is not None for p in editor.parameters() if p.requires_grad)

    # 2. Forced k-step online rollout + endpoint loss
    cfg_roll = dict(cfg, edit_prompt_dropout=0.0, rollout_prob=1.0, terminal_rollout_prob=0.0, rollout_cfg_prob=1.0)
    editor.zero_grad(set_to_none=True)
    loss_r, fields_r = _update(editor, sample_base, cfg_roll, grid, random.Random(7))
    assert fields_r["mode"] == "rollout_step"
    assert 0 <= fields_r["rollout_step"] < int(cfg["edit_steps"])
    loss_r.backward()
    assert any(p.grad is not None for p in editor.parameters() if p.requires_grad)

    # 3. Forced K-step terminal rollout
    cfg_term = dict(cfg, edit_prompt_dropout=0.0, rollout_prob=0.0, terminal_rollout_prob=1.0)
    editor.zero_grad(set_to_none=True)
    loss_k, fields_k = _update(editor, sample_base, cfg_term, grid, random.Random(11))
    assert fields_k["mode"] == "rollout_terminal"
    assert fields_k["rollout_step"] == int(cfg["edit_steps"])
    loss_k.backward()

    # 4. Standalone tail update (both tail_edited from q_K and tail_native)
    cfg_tail = _make_v2_cfg(component="tail", parameterization="gen_lora")
    tail = TailGenerator(rt, cfg_tail).train()
    tail_sample = dict(
        uid="t1", role="tail_pair", source_kind="tail_pair",
        source=sample_base["source"], target=sample_base["target"],
        tail_state=torch.randn(1, 3, 16, 16), valid=valid,
        edit_request="add one apple", tail_condition="two apples on a table",
        is_pure_t2i=False, task="T8", semantic_task="count", count_direction="add",
    )
    tail.zero_grad(set_to_none=True)
    cfg_tail_ed = dict(cfg_tail, tail_native_prob=0.0, tail_prompt_dropout=0.0)
    l_te, f_te = _tail_update(tail, tail_sample, cfg_tail_ed, None, random.Random(1))
    assert f_te["mode"] == "tail_edited"
    l_te.backward()

    tail.zero_grad(set_to_none=True)
    cfg_tail_nat = dict(cfg_tail, tail_native_prob=0.999999, tail_prompt_dropout=0.0)
    l_tn, f_tn = _tail_update(tail, tail_sample, cfg_tail_nat, None, random.Random(2))
    assert f_tn["mode"] == "tail_native"
    l_tn.backward()


def test_multi_manifest_and_tail_sampler(tmp_path):
    """Multiple manifests merge; every record is traversed once at equal weight."""
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(component="editor")
    cfg.pop("task_weights", None)
    cfg["keep_fraction"] = 0.2

    src = tmp_path / "src.png"
    src_val = tmp_path / "src_val.png"
    dst = tmp_path / "dst.png"
    Image.new("RGB", (16, 16), (120, 40, 20)).save(src)
    Image.new("RGB", (16, 16), (10, 90, 50)).save(src_val)
    Image.new("RGB", (16, 16), (40, 120, 200)).save(dst)

    rows_a = [
        dict(uid="c1", split="train", task="T8", source_image=str(src), target_image=str(dst),
             edit_instruction="add one cat", tail_condition="a cat on a chair"),
        dict(uid="s1", split="train", task="T9", source_image=str(src), target_image=str(dst),
             edit_instruction="move cat to the left", tail_condition="a cat at the left edge"),
    ]
    rows_b = [
        dict(uid="o1", split="train", task="T2", source_image=str(src), target_image=str(dst),
             edit_instruction="change wall to blue", tail_condition="a blue wall"),
        dict(uid="v1", split="val", task="T8", source_image=str(src_val), target_image=str(dst),
             edit_instruction="remove one dog", tail_condition="an empty room"),
    ]
    p1 = tmp_path / "m1.json"
    p2 = tmp_path / "m2.json"
    p1.write_text(json.dumps({"records": rows_a}))
    p2.write_text(json.dumps({"records": rows_b}))

    manifest = LinearManifest([p1, p2], rt, cfg)
    assert manifest.count == 3
    sampler = TrainingSampler(manifest, cfg, rank=0)
    drawn = {d["uid"] for d in (sampler.draw() for _ in range(3))}
    assert drawn == {"c1", "s1", "o1"}
    assert classify_semantic_task("Add 1 apple", None) == ("count", "add")
    assert classify_semantic_task("Remove the third cup", "T8") == ("count", "remove")

    # A task mixture must be refused now that traversal is record-equal.
    with pytest.raises(ValueError, match="task_weights is retired"):
        TrainingSampler(manifest, dict(cfg, task_weights={"T8": 0.7}), rank=0)

    state_pt = tmp_path / "q_k.pt"
    torch.save(torch.randn(3, 16, 16), state_pt)
    tail_manifest_path = tmp_path / "tail.json"
    tail_manifest_path.write_text(json.dumps({
        "records": [
            dict(uid="t1", split="train", task="T8", target_image=str(dst),
                 tail_state_path="q_k.pt", tail_condition="three apples"),
        ]
    }))
    cfg_tail = _make_v2_cfg(component="tail")
    tm = TailManifest(tail_manifest_path, rt, cfg_tail)
    ts = TailSampler(tm, cfg_tail, rank=0)
    ts_sample = ts.draw()
    assert ts_sample["tail_state"].shape == (1, 3, 16, 16)


def test_inference_cfg_and_tail_generator(tmp_path, monkeypatch):
    torch.manual_seed(5)
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(component="editor")
    img_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), (64, 128, 192)).save(img_path)

    editor = StateEditor(rt, cfg).eval()
    tail = TailGenerator(rt, _make_v2_cfg(component="tail")).eval()
    images, states, record = edit_pair(
        rt, editor, cfg, str(img_path), "add two stars", 11,
        edit_cfg_scale=1.5, tail=tail,
    )
    assert "result" in images and "result" in states
    assert record["edit_cfg_scale"] == 1.5
    assert record["tail_generator"] is True

    monkeypatch.setattr(
        inference_mod,
        "review_reflection",
        lambda *a, **kw: {
            "reflection": {
                "outcome": "edit",
                "instruction": "add two stars",
                "match": "no",
                "description": "no stars",
                "preserve": "",
                "raw": "edit: add two stars",
            }
        },
    )
    images_t2i, edited_t2i, record_t2i = t2i_feedback(
        rt, editor, cfg, "two stars", 13, edit_cfg_scale=1.5, tail=tail,
    )
    assert edited_t2i is not None
    assert "result" in images_t2i and "edited_preview" in images_t2i
    assert record_t2i["reflection"]["outcome"] == "edit"


def test_tail_field_passes_string_prompt_to_tail_t2i():
    seen = []
    rt = _DummyRuntime()
    orig_prefix = rt.prefix

    def checked_prefix(prompt):
        seen.append(prompt)
        return orig_prefix(prompt)

    rt.prefix = checked_prefix
    tail = TailGenerator(rt, _make_v2_cfg(component="tail")).eval()
    field = inference_mod._tail_field(rt, "a red apple", _make_v2_cfg(component="editor"), tail=tail)
    assert isinstance(field, TailT2I)
    assert seen == ["a red apple", None, "a red apple", None]


def test_real_configs_validate():
    for name in (
        "stage1_full490k.json",
        "editor_v2_rollout_lora.json",
        "editor_v2_rollout_gen_full.json",
        "tail_v1_lora.json",
    ):
        cfg = load_config(CONFIG_DIR / name)
        validate_config(cfg)


def test_linear_manifest_requires_a_target_caption(tmp_path):
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(component="editor")
    src, dst = tmp_path / "s.png", tmp_path / "d.png"
    Image.new("RGB", (16, 16)).save(src)
    Image.new("RGB", (16, 16)).save(dst)
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"records": [
        dict(uid="nocap", split="train", source_image=str(src), target_image=str(dst),
             edit_instruction="add one cat"),
    ]}))
    with pytest.raises(ValueError, match="target caption"):
        LinearManifest(path, rt, cfg)


def test_target_captions_fill_the_tail_condition(tmp_path):
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(component="editor")
    src, dst = tmp_path / "s.png", tmp_path / "d.png"
    Image.new("RGB", (16, 16)).save(src)
    Image.new("RGB", (16, 16)).save(dst)
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"records": [
        dict(uid="capme", split="train", source_image=str(src), target_image=str(dst),
             edit_instruction="add one cat"),
    ]}))
    captions = tmp_path / "caps.jsonl"
    captions.write_text(json.dumps({"uid": "capme", "parent_condition": "two cats on a sofa"}) + "\n")
    manifest = LinearManifest(path, rt, cfg, captions=[captions])
    assert manifest.rows[0]["tail_condition"] == "two cats on a sofa"
    assert manifest.captions_paths == [captions.resolve()]
    assert manifest.sample(0, 0.36)["tail_condition"] == "two cats on a sofa"


def test_derived_tails_are_gated_by_the_configuration(tmp_path):
    rt = _DummyRuntime()
    src, dst = tmp_path / "s.png", tmp_path / "d.png"
    Image.new("RGB", (16, 16)).save(src)
    Image.new("RGB", (16, 16)).save(dst)
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"records": [
        dict(uid="derived", split="train", source_image=str(src), target_image=str(dst),
             edit_instruction="add one cat", tail_condition="a cat",
             condition_protocol="derived_parent_plus_edit"),
    ]}))
    with pytest.raises(ValueError, match="Derived parent"):
        LinearManifest(path, rt, _make_v2_cfg(component="editor", allow_derived_tail=False))
    manifest = LinearManifest(path, rt, _make_v2_cfg(component="editor", allow_derived_tail=True))
    assert manifest.rows[0]["tail_source"] == "derived_parent_plus_edit"


def test_keep_draws_take_the_identity_path_not_the_paired_edit():
    torch.manual_seed(3)
    rt = _DummyRuntime()
    cfg = _make_v2_cfg(component="editor")
    grid = phase_grid(rt, cfg)
    editor = StateEditor(rt, cfg).train()
    source = torch.randn(1, 3, 16, 16)
    keep_sample = dict(
        uid="k1", role="L_edit", source_kind="linear_pair",
        source=source, target=source.clone(), valid=torch.ones(1, 1, 16, 16),
        edit_request=cfg["keep_instruction"], image_prompt="", keep=True, time=0.36,
    )
    # Rollouts must not be scheduled for a no-op edit.
    cfg_roll = dict(cfg, rollout_prob=1.0, terminal_rollout_prob=1.0, edit_prompt_dropout=1.0)
    editor.zero_grad(set_to_none=True)
    loss_keep, fields_keep = _update(editor, keep_sample, cfg_roll, grid, random.Random(0))
    assert fields_keep["mode"] == "keep"
    assert fields_keep["keep"] == 1
    assert fields_keep["prompt_dropped"] == 0
    assert fields_keep["progress"] >= 0.5
    loss_keep.backward()
    assert any(p.grad is not None for p in editor.parameters() if p.requires_grad)
    assert torch.isfinite(loss_keep)
