"""IsochronicTransitionV2: model-agnostic conditional-anchor manifold transition operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .contracts import AnchorMode, ConditionBundle, LatentState, TransitionResult


@dataclass
class TransitionConfig:
    read_progress: float = 0.36
    stop_progress: float = 0.60
    output_progress: float | None = None
    anchor_mode: AnchorMode = "conditional_native"
    edit_cfg_scale: float = 1.4
    trust_region_rho: float = 0.65
    noise_refresh: float = 0.0
    correction_steps: int = 1
    num_rollout_steps: int = 8


class IsochronicTransitionV2:
    """Executes short-horizon semantic correction + conditional-anchor isochronic reprojection.

    Key upgrades over legacy isochronic_manifold_v1 (grounded in Step 1 ablation):
      1. Defaults to conditional native anchor H_0(z_{p0}, p0, original_prompt), cutting
         GenEval2 add Damage by >50% with zero extra NFE when prefix clean estimate is cached.
      2. Applies CorrFlow-style RMS trust-region clamping on edit CFG displacement delta so
         moderate edit_cfg_scale (1.3-1.5) breaks MMSE mode collapse without OOB blowup.
      3. Supports optional K-step fixed-anchor refinement (Correction Training style).
    """

    def __init__(self, config: TransitionConfig | None = None):
        self.config = config or TransitionConfig()

    def _rollout_grid(
        self,
        generator_adapter,
        p0: float,
        p_stop: float,
    ) -> list[float]:
        grid = generator_adapter.progress_grid()
        interior = [float(p) for p in grid if p0 + 1e-6 < float(p) < p_stop - 1e-6]
        if interior:
            return [p0] + interior + [p_stop]
        steps = max(1, int(self.config.num_rollout_steps))
        return [p0 + (p_stop - p0) * i / steps for i in range(steps + 1)]

    @torch.no_grad()
    def _guided_clean(
        self,
        generator_adapter,
        z: torch.Tensor,
        p: float,
        edit_cond: ConditionBundle,
        null_cond: ConditionBundle,
        x_ref: torch.Tensor,
    ) -> torch.Tensor:
        x_cond = generator_adapter.predict_clean(z, p, edit_cond, use_corrector=True)
        scale = float(self.config.edit_cfg_scale)
        if scale <= 1.0 + 1e-6:
            return x_cond
        x_null = generator_adapter.predict_clean(z, p, null_cond, use_corrector=True)
        delta = x_cond.float() - x_null.float()
        rho = float(self.config.trust_region_rho)
        if rho > 0.0:
            delta = generator_adapter.geometry.trust_region_clip(delta, x_ref, rho)
        return x_null.float() + scale * delta

    @torch.no_grad()
    def apply(
        self,
        generator_adapter,
        state: LatentState,
        edit_cond: ConditionBundle,
        original_cond: ConditionBundle,
        null_cond: ConditionBundle,
        *,
        cached_native_clean: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> TransitionResult:
        cfg = self.config
        geom = generator_adapter.geometry
        p0 = float(state.progress)
        p_stop = float(cfg.stop_progress)
        p_out = p0 if cfg.output_progress is None else float(cfg.output_progress)
        if not 0.0 < p0 < p_stop < 1.0:
            raise ValueError(f"Expected 0 < p0 ({p0}) < stop_progress ({p_stop}) < 1")
        if not p0 - 1e-6 <= p_out < 1.0:
            raise ValueError(f"Expected p0 ({p0}) <= output_progress ({p_out}) < 1")

        # 1. Extract structural noise anchor (conditional native default, zero extra NFE if cached)
        anchor_nfe = 0
        if cfg.anchor_mode == "conditional_native":
            if cached_native_clean is not None:
                x_source_est = cached_native_clean.detach().float()
            else:
                x_source_est = generator_adapter.predict_clean(
                    state.tensor.detach(), p0, original_cond, use_corrector=False
                )
                anchor_nfe = 1
        elif cfg.anchor_mode == "empty_native":
            x_source_est = generator_adapter.predict_clean(
                state.tensor.detach(), p0, null_cond, use_corrector=False
            )
            anchor_nfe = 1
        else:
            raise ValueError(f"Unsupported anchor_mode: {cfg.anchor_mode}")

        eps_anchor = geom.extract_eps_anchor(state.tensor, p0, x_source_est)
        times = self._rollout_grid(generator_adapter, p0, p_stop)

        # 2. Short-horizon adapted rollout (optionally repeated K times keeping eps_anchor fixed)
        z_curr = state.tensor.detach().float()
        clean_pred = x_source_est
        rollout_calls = 0
        k_steps = max(1, int(cfg.correction_steps))
        for k_idx in range(k_steps):
            q = z_curr.clone()
            for left, right in zip(times[:-1], times[1:]):
                x_est = self._guided_clean(generator_adapter, q, left, edit_cond, null_cond, x_source_est)
                rollout_calls += 2 if cfg.edit_cfg_scale > 1.0 + 1e-6 else 1
                v_clean = geom.from_clean_to_v_clean(q, left, x_est)
                q = q + (right - left) * v_clean
            clean_pred = self._guided_clean(generator_adapter, q, p_stop, edit_cond, null_cond, x_source_est)
            rollout_calls += 2 if cfg.edit_cfg_scale > 1.0 + 1e-6 else 1
            if k_idx + 1 < k_steps:
                z_curr = geom.compose_state(
                    clean_pred,
                    eps_anchor,
                    p0,
                    eta=0.0,
                    generator=generator,
                )

        # 3. Reproject onto native noisy manifold at p_out
        z_out = geom.compose_state(
            clean_pred,
            eps_anchor,
            p_out,
            eta=float(cfg.noise_refresh),
            generator=generator,
        )
        edit_rms = float((clean_pred.float() - x_source_est.float()).square().mean().sqrt().item())
        out_state = LatentState(
            tensor=z_out,
            progress=p_out,
            valid_mask=state.valid_mask,
            meta=dict(state.meta, transition="isochronic_v2"),
        )
        return TransitionResult(
            state=out_state,
            clean_target_est=clean_pred,
            clean_source_est=x_source_est,
            eps_anchor=eps_anchor,
            telemetry={
                "input_progress": p0,
                "stop_progress": p_stop,
                "output_progress": p_out,
                "anchor_mode": cfg.anchor_mode,
                "edit_cfg_scale": float(cfg.edit_cfg_scale),
                "trust_region_rho": float(cfg.trust_region_rho),
                "noise_refresh": float(cfg.noise_refresh),
                "correction_steps": k_steps,
                "anchor_nfe": anchor_nfe,
                "correction_nfe": rollout_calls,
                "edit_clean_rms": round(edit_rms, 5),
                "rollout_times": times,
            },
        )
