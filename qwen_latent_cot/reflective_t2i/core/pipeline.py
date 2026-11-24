"""Model-agnostic ReflectiveT2IPipeline supporting all 5 causal experiment arms (A-E)."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Literal

import torch

from .contracts import (
    ConditionBundle,
    ConditionMode,
    ObserverBackend,
    PipelineResult,
    ReflectionDecision,
)
from .correction import IsochronicTransitionV2, TransitionConfig
from .reflection_policy import ReflectionPolicy, ReflectionPolicyConfig, compile_concise_imperative


ExperimentArm = Literal[
    "A_native",
    "B_rewrite_resample",
    "C_reflective",
    "D_oracle_feedback",
    "E_shuffled_feedback",
]


@dataclass
class PipelineConfig:
    read_progress: float = 0.36
    stop_progress: float = 0.60
    lookahead_steps: int = 4
    guidance_scale: float = 4.0
    cfg_interval: tuple[float, float] = (0.0, 0.80)
    max_corrections: int = 1
    condition_mode: ConditionMode = "feedback_only"
    concise_imperative: bool = True
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    reflection: ReflectionPolicyConfig = field(default_factory=ReflectionPolicyConfig)


class ReflectiveT2IPipeline:
    """Unified Generate -> Preview -> Observe -> Reflect -> Correct -> Continue loop."""

    def __init__(
        self,
        generator_adapter,
        observer: ObserverBackend | None,
        config: PipelineConfig | None = None,
    ):
        self.generator = generator_adapter
        self.observer = observer
        self.config = config or PipelineConfig()
        self.policy = ReflectionPolicy(self.config.reflection)
        self.transition = IsochronicTransitionV2(self.config.transition)

    def _build_edit_condition(
        self,
        original_prompt: str,
        decision: ReflectionDecision,
    ) -> ConditionBundle:
        if self.config.concise_imperative:
            instruction = compile_concise_imperative(
                decision.instruction,
                action=decision.action,
                coarse_region_hint=decision.coarse_region_hint,
            )
        else:
            instruction = decision.instruction
            if decision.coarse_region_hint and decision.coarse_region_hint.lower() not in instruction.lower():
                instruction = f"{instruction.rstrip('.')} ({decision.coarse_region_hint})."
        return self.generator.encode_edit_condition(
            feedback=instruction,
            original_prompt=original_prompt,
            target_rewrite=decision.target_rewrite or original_prompt,
            condition_mode=self.config.condition_mode,
        )

    @torch.no_grad()
    def run(
        self,
        prompt: str,
        *,
        seed: int,
        arm: ExperimentArm = "C_reflective",
        oracle_instruction: str | None = None,
        shuffled_instruction: str | None = None,
    ) -> PipelineResult:
        t_start = time.perf_counter()
        cfg = self.config
        p0 = float(cfg.read_progress)

        orig_cond = self.generator.encode_prompt(prompt)
        null_cond = self.generator.encode_prompt(None)

        z_init = self.generator.sample_initial_noise(seed=seed)

        # Arm A: Native T2I baseline (exact shared prefix + tail)
        if arm == "A_native":
            state_p0, _ = self.generator.sample_until(
                z_init,
                p0,
                orig_cond,
                null_cond,
                cfg.guidance_scale,
                cfg_interval=cfg.cfg_interval,
                return_clean_at_end=False,
            )
            final_tensor = self.generator.continue_native(
                state_p0,
                orig_cond,
                null_cond,
                cfg.guidance_scale,
                cfg_interval=cfg.cfg_interval,
            )
            return PipelineResult(
                image=self.generator.decode_preview(final_tensor),
                final_tensor=final_tensor,
                arm=arm,
                telemetry={"wall_seconds": round(time.perf_counter() - t_start, 4), "triggered": False},
            )

        # Shared native prefix 0 -> p0 (caching native clean estimate at p0 for zero-cost conditional anchor)
        state, cached_clean = self.generator.sample_until(
            z_init,
            p0,
            orig_cond,
            null_cond,
            cfg.guidance_scale,
            cfg_interval=cfg.cfg_interval,
            return_clean_at_end=True,
        )

        previews = []
        decisions: list[ReflectionDecision] = []
        transitions = []

        # Arm B: Outer-loop VLM + Prompt Rewrite + Resample (same seed/noise budget baseline)
        if arm == "B_rewrite_resample":
            if self.observer is None:
                raise ValueError("Arm B_rewrite_resample requires an ObserverBackend")
            preview = self.generator.lookahead_preview(
                state, orig_cond, null_cond, cfg.guidance_scale, steps=cfg.lookahead_steps
            )
            previews.append(preview)
            decision = self.policy.reflect(self.observer, preview, prompt)
            decisions.append(decision)
            if decision.outcome == "edit" and decision.target_rewrite:
                rewritten_cond = self.generator.encode_prompt(decision.target_rewrite)
                state_rw, _ = self.generator.sample_until(
                    z_init.clone(),
                    p0,
                    rewritten_cond,
                    null_cond,
                    cfg.guidance_scale,
                    cfg_interval=cfg.cfg_interval,
                    return_clean_at_end=False,
                )
                final_tensor = self.generator.continue_native(
                    state_rw,
                    rewritten_cond,
                    null_cond,
                    cfg.guidance_scale,
                    cfg_interval=cfg.cfg_interval,
                )
            else:
                final_tensor = self.generator.continue_native(
                    state, orig_cond, null_cond, cfg.guidance_scale, cfg_interval=cfg.cfg_interval
                )
            return PipelineResult(
                image=self.generator.decode_preview(final_tensor),
                final_tensor=final_tensor,
                preview_images=previews,
                decisions=decisions,
                arm=arm,
                telemetry={
                    "wall_seconds": round(time.perf_counter() - t_start, 4),
                    "triggered": decision.outcome == "edit",
                },
            )

        # Arms C / D / E: Isochronic state correction loop (1..max_corrections rounds)
        for round_idx in range(max(1, int(cfg.max_corrections))):
            if arm == "D_oracle_feedback":
                if not oracle_instruction:
                    raise ValueError("Arm D_oracle_feedback requires oracle_instruction")
                decision = ReflectionDecision(
                    outcome="edit",
                    action="other",
                    instruction=oracle_instruction,
                    target_rewrite=prompt,
                    confidence=1.0,
                )
            elif arm == "E_shuffled_feedback":
                if not shuffled_instruction:
                    raise ValueError("Arm E_shuffled_feedback requires shuffled_instruction")
                decision = ReflectionDecision(
                    outcome="edit",
                    action="other",
                    instruction=shuffled_instruction,
                    target_rewrite=prompt,
                    confidence=1.0,
                )
            else:
                if self.observer is None:
                    raise ValueError("Arm C_reflective requires an ObserverBackend")
                preview = self.generator.lookahead_preview(
                    state, orig_cond, null_cond, cfg.guidance_scale, steps=cfg.lookahead_steps
                )
                previews.append(preview)
                decision = self.policy.reflect(self.observer, preview, prompt)

            decisions.append(decision)
            if decision.outcome != "edit" or not decision.instruction:
                break

            edit_cond = self._build_edit_condition(prompt, decision)
            tr_gen = torch.Generator(device=state.tensor.device).manual_seed(int(seed) * 97 + round_idx)
            tr = self.transition.apply(
                self.generator,
                state,
                edit_cond,
                orig_cond,
                null_cond,
                cached_native_clean=cached_clean if round_idx == 0 else None,
                generator=tr_gen,
            )
            transitions.append(tr)
            state = tr.state
            cached_clean = None  # Subsequent round re-evaluates native anchor on updated state
            if arm in ("D_oracle_feedback", "E_shuffled_feedback"):
                break

        final_tensor = self.generator.continue_native(
            state,
            orig_cond,
            null_cond,
            cfg.guidance_scale,
            cfg_interval=cfg.cfg_interval,
        )
        return PipelineResult(
            image=self.generator.decode_preview(final_tensor),
            final_tensor=final_tensor,
            preview_images=previews,
            decisions=decisions,
            transitions=transitions,
            arm=arm,
            telemetry={
                "wall_seconds": round(time.perf_counter() - t_start, 4),
                "triggered": len(transitions) > 0,
                "num_corrections": len(transitions),
            },
        )
