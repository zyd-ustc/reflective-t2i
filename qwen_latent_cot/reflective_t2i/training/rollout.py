"""Hybrid mid-band bridge + HY-SOAR 1-step self-rollout training state builder.

Solves two core failure modes diagnosed in Step 1:
  1. Target leakage & exposure bias on `add/count`: 40% of updates start strictly from
     zero-leakage anchor a_{p0}, take 1 detached on-policy rollout step (ODE or CPS
     stochastic step) using the corrector itself to produce a realistic off-manifold state
     q'_{p1}, and supervise H_theta(q'_{p1}, p1, e) -> T.
  2. Corrupted edit CFG null branch: dropped-prompt rows (`edit_prompt_dropout`) anchor on
     a_{p0} with `target = source` (identity/no-op branch) instead of random target T, so
     edit CFG extrapolates cleanly along the semantic edit direction.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from ..core.contracts import ConditionBundle
from ..core.noise_geometry import LinearFlowGeometry


@dataclass
class RolloutConfig:
    read_progress: float = 0.36
    train_max_progress: float = 0.60
    bridge_start_prob: float = 0.25
    soar_rollout_prob: float = 0.15
    soar_warmup_steps: int = 0
    soar_cps_prob: float = 0.0
    soar_cps_eta: float = 0.30
    edit_prompt_dropout: float = 0.10
    dropout_target: str = "source_identity"  # "source_identity" (recommended) or "target_legacy"


def build_training_batch_step(
    generator_adapter,
    source_clean: torch.Tensor,
    target_clean: torch.Tensor,
    edit_cond: ConditionBundle,
    null_cond: ConditionBundle,
    config: RolloutConfig,
    *,
    step: int = 1,
    rng: random.Random | None = None,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float, ConditionBundle, torch.Tensor, dict[str, Any]]:
    """Construct `(state, progress, active_cond, supervision_target, stats)` for one step."""
    py_rng = rng or random
    geom: LinearFlowGeometry = generator_adapter.geometry
    p0 = float(config.read_progress)
    p_max = float(config.train_max_progress)
    if not 0.0 < p0 < p_max < 1.0:
        raise ValueError(f"Expected 0 < read_progress ({p0}) < train_max_progress ({p_max}) < 1")

    eps = geom.sample_noise(
        tuple(source_clean.shape),
        device=source_clean.device,
        dtype=torch.float32,
        generator=torch_generator,
    )
    anchor = geom.interpolate(source_clean, eps, p0)

    # 1. Prompt dropout branch (identity anchor across the FULL rollout window [p0, p_max] for clean edit CFG)
    dropped = config.edit_prompt_dropout > 0.0 and py_rng.random() < float(config.edit_prompt_dropout)
    if dropped:
        tau = p0 if py_rng.random() < config.bridge_start_prob else p0 + (p_max - p0) * py_rng.random()
        if config.dropout_target == "source_identity":
            state_id = anchor if tau == p0 else geom.interpolate(source_clean, eps, tau)
            return (
                state_id,
                tau,
                null_cond,
                source_clean.float(),
                {
                    "branch": "dropout_identity",
                    "progress": tau,
                    "prompt_dropped": 1,
                    "soar_used": 0,
                },
            )
        alpha = (tau - p0) / (1.0 - p0)
        state = (1.0 - alpha) * anchor + alpha * target_clean.float()
        return (
            state,
            tau,
            null_cond,
            target_clean.float(),
            {"branch": "dropout_legacy", "progress": tau, "prompt_dropped": 1, "soar_used": 0},
        )

    # 2. HY-SOAR 1-step on-policy self-rollout branch (activated after soar_warmup_steps)
    soar_enabled = int(step) > int(config.soar_warmup_steps)
    use_soar = soar_enabled and config.soar_rollout_prob > 0.0 and py_rng.random() < float(config.soar_rollout_prob)
    if use_soar:
        p1 = p0 + (p_max - p0) * max(0.05, py_rng.random())
        with torch.no_grad():
            x_pred_0 = generator_adapter.predict_clean(anchor, p0, edit_cond, use_corrector=True)
            use_cps = config.soar_cps_prob > 0.0 and py_rng.random() < float(config.soar_cps_prob)
            if use_cps:
                q_next = geom.cps_step(
                    anchor,
                    p0,
                    p1,
                    x_pred_0,
                    eta=config.soar_cps_eta,
                    generator=torch_generator,
                ).detach()
                branch = "soar_cps"
            else:
                v_0 = geom.from_clean_to_v_clean(anchor, p0, x_pred_0)
                q_next = (anchor + (p1 - p0) * v_0).detach()
                branch = "soar_ode"
        return (
            q_next,
            p1,
            edit_cond,
            target_clean.float(),
            {"branch": branch, "progress": p1, "prompt_dropped": 0, "soar_used": 1},
        )

    # 3. Standard mid-band bridge branch with elevated p0 entry coverage
    if py_rng.random() < float(config.bridge_start_prob):
        tau = p0
        state = anchor
        branch = "bridge_entry"
    else:
        tau = p0 + (p_max - p0) * py_rng.random()
        alpha = (tau - p0) / (1.0 - p0)
        state = (1.0 - alpha) * anchor + alpha * target_clean.float()
        branch = "bridge_mid"

    return (
        state,
        tau,
        edit_cond,
        target_clean.float(),
        {"branch": branch, "progress": tau, "prompt_dropped": 0, "soar_used": 0},
    )
