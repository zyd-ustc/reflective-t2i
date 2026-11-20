"""Unit tests for the unified Strict Isochronic Manifold Operator + Breakthrough A."""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from qwen_latent_cot.u15_target_correction import (
    CONDITION_PROTOCOL,
    CURRENT_GEOMETRY_PROTOCOL,
    EDITOR_INTERFACE,
    PARAMETER_LAYOUT,
)
from qwen_latent_cot.u15_target_correction.checkpoint import (
    canonical_parameters,
    editor_identity,
    load_stage_weights,
    resume,
    save,
)
from qwen_latent_cot.u15_target_correction.conditions import feedback_prompt
from qwen_latent_cot.u15_target_correction.config import load_config, validate_config
from qwen_latent_cot.u15_target_correction.diagnosis import (
    check_instruction_consistency,
    extract_add_subject,
    review_reflection,
)
from qwen_latent_cot.u15_target_correction.field import (
    isochronic_manifold_transition,
    phase_grid,
    preview_image,
    preview_subgrid,
    project_clean_to_native_manifold,
    segment_times,
)
from qwen_latent_cot.u15_target_correction.loss import clean_target_mse
from qwen_latent_cot.u15_target_correction.model import IsochronicManifoldEditor, build_editor
from qwen_latent_cot.u15_target_correction.train import isochronic_manifold_update

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "u15_isochronic_state" / "isochronic_manifold.json"


class DummyRuntime:
    """Lightweight CPU SenseNova-U1.5 stub for unit testing."""

    def __init__(self, width=16, height=16, hidden=8):
        self.width = width
        self.height = height
        self.patch = 8
        self.grid = (height // 8, width // 8)
        self.tokens = self.grid[0] * self.grid[1]
        self.device = torch.device("cpu")
        self.identity = {"stub": "u15"}

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                for n in ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen"):
                    setattr(self, n, nn.Linear(hidden, hidden))

            def forward(self, x):
                return self.o_proj_mot_gen(
                    self.q_proj_mot_gen(x) + self.k_proj_mot_gen(x) + self.v_proj_mot_gen(x)
                )

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(hidden, hidden)
                self.up_proj = nn.Linear(hidden, hidden)
                self.down_proj = nn.Linear(hidden, hidden)

            def forward(self, x):
                return self.down_proj(torch.tanh(self.gate_proj(x)) + self.up_proj(x))

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _Attn()
                self.mlp_mot_gen = _MLP()

            def forward(self, hidden_states, *, past_key_values=None, **kwargs):
                bias = float(past_key_values) * 0.05 if past_key_values is not None else 0.0
                return hidden_states + 0.1 * (self.self_attn(hidden_states) + self.mlp_mot_gen(hidden_states)) + bias

        class _TimeEmbed(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(1, hidden)

            def forward(self, t):
                return self.proj(t.float().unsqueeze(-1))

        self.decoder = SimpleNamespace(
            layers=nn.ModuleList([_Layer(), _Layer()]),
            norm_mot_gen=nn.LayerNorm(hidden),
        )
        vit_patch = SimpleNamespace(
            embeddings=SimpleNamespace(patch_embedding=SimpleNamespace(weight=torch.zeros(1, dtype=torch.float32)))
        )
        fm_head = nn.Linear(hidden, 3 * self.patch * self.patch)
        self.model = SimpleNamespace(
            eval=lambda: self.model,
            requires_grad_=lambda flag: self.model,
            patch_size=self.patch,
            add_noise_scale_embedding=False,
            noise_scale_max_value=16.0,
            use_pixel_head=False,
            use_deep_fm_head=False,
            config=SimpleNamespace(t_eps=0.05),
            fm_modules=nn.ModuleDict(
                {
                    "vision_model_mot_gen": nn.Module(),
                    "timestep_embedder": _TimeEmbed(),
                    "fm_head": fm_head,
                }
            ),
            patchify=lambda z, p, channel_first=True: z.reshape(z.shape[0], 3, self.grid[0], p, self.grid[1], p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(z.shape[0], self.tokens, 3 * p * p),
            unpatchify=lambda c, p, h, w: c.reshape(c.shape[0], h // p, w // p, 3, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(c.shape[0], 3, h, w),
            extract_feature=lambda flat, gen_model=True, grid_hw=None: flat[:, :hidden],
        )
        object.__setattr__(self.model.fm_modules["vision_model_mot_gen"], "embeddings", vit_patch.embeddings)

    def noise_scale(self):
        return 1.0

    def times(self):
        return [round(i / 20.0, 4) for i in range(21)]

    def prefix(self, prompt):
        bias = 0.0 if prompt is None else float(len(str(prompt)) % 7 + 1)
        return SimpleNamespace(cache=bias, indexes=torch.zeros(1))


def make_cfg(**overrides):
    cfg = load_config(CONFIG_PATH)
    cfg.update(overrides)
    return cfg


def test_canonical_config_and_retired_key_rejection():
    cfg = load_config(CONFIG_PATH)
    validate_config(cfg)
    assert cfg["editor_interface"] == EDITOR_INTERFACE
    assert cfg["condition_protocol"] == CONDITION_PROTOCOL
    assert cfg["read_time"] == 0.36
    assert cfg["stop_time"] == 0.60
    assert cfg["train_max_time"] == 0.60
    assert cfg["lookahead_steps"] == 4
    assert cfg["recheck_add_on_preview"] is True
    assert cfg["gate_contradictory_edits"] is True
    assert cfg["cfg_interval"] == [0.0, 0.8]

    # Ensure no legacy config files remain
    cfg_dir = CONFIG_PATH.parent
    assert sorted(p.name for p in cfg_dir.glob("*.json")) == ["core.json", "isochronic_manifold.json"]

    for bad in (
        {"edit_steps": 8},
        {"rollout_prob": 0.5},
        {"tail_t_min": 0.36},
        {"lambda_distill": 0.5},
        {"component": "tail"},
        {"lookahead_steps": 1},
        {"stop_time": 0.36},
        {"stop_time": 1.0},
    ):
        with pytest.raises(ValueError):
            validate_config({**cfg, **bad})


def test_breakthrough_a_multistep_preview_and_reflection_gates(monkeypatch):
    rt = DummyRuntime(width=16, height=16)
    cfg = make_cfg(width=16, height=16, lora_rank=4, activation_checkpointing=False)
    grid = phase_grid(rt, cfg)

    sub2 = preview_subgrid(grid, 0.36, steps=2)
    sub4 = preview_subgrid(grid, 0.36, steps=4)
    assert len(sub2) == 3 and sub2[0] == 0.36 and sub2[-1] == 1.0
    assert len(sub4) == 5 and sub4[0] == 0.36 and sub4[-1] == 1.0
    assert 0.36 < sub4[1] < sub2[1]  # 4-step preview resolves the [0.36, 0.60] emergence band!

    state = torch.randn(1, 3, 16, 16)
    before = state.clone()
    calls = []

    def vel(z, t):
        calls.append(float(t))
        return -0.2 * z

    prev4 = preview_image(state, 0.36, grid, vel, steps=4)
    torch.testing.assert_close(state, before)
    assert prev4.shape == state.shape
    assert len(calls) == 5  # 2 for Heun on first interval + 3 Euler intervals

    # Gate 2 deterministic checks on real failure cases observed on GenEval / GenEval++:
    assert extract_add_subject(
        "Add 1 cell phone to the image so that it contains exactly 1 cell phone instead of the current 0"
    ) == "cell phone"

    # Case f8c601dd: prompt wanted 1 green hot dog, UND hallucinated "Add 2 ... contains 3 instead of 1"
    ok, reason = check_instruction_consistency(
        "Add 2 green hot dogs so that it contains exactly 3 green hot dogs instead of the current 1, placing the new ones in plausible empty positions; keep everything else unchanged.",
        "A photo of three white boats and one green hot dogs",
    )
    assert ok is False and reason == "current_count_already_matches_prompt"

    # Case counting_0208: prompt wanted three zebras, UND wrote "Remove 3 zebras ... contains 0 zebras"
    ok2, reason2 = check_instruction_consistency(
        "Remove 3 zebras from the background so that it contains exactly 0 zebras in the background instead of the current 3; keep everything else unchanged.",
        "a photo of three zebras",
    )
    assert ok2 is False and reason2 == "removes_required_subject_to_zero"

    # Valid rescue: prompt wanted 4 keyboards, current is 6
    ok3, reason3 = check_instruction_consistency(
        "Remove 2 black keyboards so that it contains exactly 4 computer keyboards instead of the current 6; keep everything else unchanged.",
        "a photo of four computer keyboards",
    )
    assert ok3 is True and reason3 is None

    # Gate 1: Prompt-blind focused add re-inspection catches overlooked small object without leaking prompt!
    vision_questions = []
    img_replies = iter(
        [
            "A brown horse standing in green grass.",
            "Looking closely at the grass near the horse's hind hooves, 1 small black cell phone is visible. Total: 1.",
        ]
    )
    txt_replies = iter(
        [
            "MATCH: no\nINSTRUCTION: Add 1 cell phone to the image so that it contains exactly 1 cell phone instead of the current 0, placing the new ones in plausible empty positions; keep everything else unchanged.",
            "MATCH: yes\nINSTRUCTION: NONE",
        ]
    )

    def fake_ask_img(_rt, _img, q, _cfg, **kw):
        vision_questions.append(q)
        return next(img_replies)

    monkeypatch.setattr("qwen_latent_cot.u15_target_correction.diagnosis.ask_with_image", fake_ask_img)
    monkeypatch.setattr("qwen_latent_cot.u15_target_correction.diagnosis.ask_text", lambda *a, **k: next(txt_replies))

    out = review_reflection(rt, prev4, "a photo of a cell phone and a horse", cfg)["reflection"]
    assert out["outcome"] == "satisfied"
    assert out["gated_by"] == "focused_add_recheck"
    assert out["rechecked_add_subject"] == "cell phone"
    assert len(vision_questions) == 2
    assert "horse" not in vision_questions[1]  # Prompt-blind: target prompt never leaked to VLM!


def test_strict_isochronic_transition_midband_training_and_step032000_compat(tmp_path):
    rt = DummyRuntime(width=16, height=16)
    cfg = make_cfg(width=16, height=16, lora_rank=4, activation_checkpointing=False)
    editor = build_editor(rt, cfg)
    assert isinstance(editor, IsochronicManifoldEditor)

    # Zero-B LoRA matches frozen native branch
    state = torch.randn(1, 3, 16, 16)
    assert feedback_prompt("Add a dog.") == "Edit request: Add a dog."
    assert feedback_prompt("", allow_empty=True) is None
    pref = editor.feedback_prefix("Add a dog.")
    torch.testing.assert_close(editor.predict_clean(state, 0.36, pref), editor._clean(state, 0.36, pref, use_adapted=False))

    # Mid-band training update samples tau strictly inside [0.36, 0.60]
    rng = random.Random(7)
    sample = {
        "source": torch.randn(1, 3, 16, 16),
        "target": torch.randn(1, 3, 16, 16),
        "valid": torch.ones(1, 1, 16, 16),
        "edit_request": "Add a dog.",
        "role": "L_edit",
    }
    seen_taus = []
    for _ in range(25):
        editor.zero_grad(set_to_none=True)
        loss, fields = isochronic_manifold_update(editor, sample, cfg, rng)
        seen_taus.append(fields["time"])
        assert 0.36 <= fields["time"] <= 0.60
    loss.backward()
    b_grads = [p.grad.abs().max().item() for n, p in canonical_parameters(editor).items() if n.endswith(".b")]
    assert max(b_grads) > 0.0

    # Strict isochronic transition preserves input_time == output_time == 0.36
    editor.eval()
    grid = phase_grid(rt, cfg)
    state_copy = state.clone()
    z_iso, clean_pred, eps_anchor, info = isochronic_manifold_transition(
        editor, state, 0.36, "Add a dog.", grid, edit_cfg_scale=1.5
    )
    torch.testing.assert_close(state, state_copy)
    assert info["input_time"] == 0.36
    assert info["output_time"] == 0.36
    assert info["stop_time"] == 0.60
    assert z_iso.shape == state.shape

    x_src = editor.native_clean(state, 0.36)
    z_id = project_clean_to_native_manifold(x_src, eps_anchor, 0.36, sigma=rt.noise_scale())
    torch.testing.assert_close(z_id, state, atol=1e-5, rtol=1e-5)

    # Checkpoint save, strict resume, AND backward-compatible weight loading of semantic_denoising_v1 (step_032000)!
    identity = editor_identity(rt, cfg)
    assert identity["parameter_layout"] == PARAMETER_LAYOUT
    assert identity["geometry"] == CURRENT_GEOMETRY_PROTOCOL
    opt = torch.optim.AdamW(canonical_parameters(editor).values(), lr=1e-4)
    sampler = SimpleNamespace(state_dict=lambda: {"c": 1}, load_state_dict=lambda s: None)
    ckpt = tmp_path / "step_032000"
    save(
        ckpt, editor, opt, sampler, 32000, cfg, identity, rt.device,
        data_version="dv1", input_domains=["linear"], total_steps=32000, budget=32000,
    )
    meta, _ = resume(ckpt, build_editor(rt, cfg), opt, sampler, cfg, identity, data_version="dv1", total_steps=32000)
    assert meta["step"] == 32000

    # Simulate step_032000 manifest written under semantic_denoising_v1
    manifest_json = json.loads((ckpt / "manifest.json").read_text())
    manifest_json["identity"]["editor_interface"] = "semantic_denoising_v1"
    (ckpt / "manifest.json").write_text(json.dumps(manifest_json))
    loaded_meta = load_stage_weights(ckpt, build_editor(rt, cfg), identity)
    assert loaded_meta["step"] == 32000
