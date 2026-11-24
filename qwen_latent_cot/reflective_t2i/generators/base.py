"""Abstract GeneratorAdapter with shared Euler/Heun flow sampling & lookahead preview."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from PIL import Image
import torch

from ..core.contracts import ConditionBundle, ConditionMode, LatentState, StateCorrector
from ..core.noise_geometry import LinearFlowGeometry


def segment_progress_grid(grid: Sequence[float], start: float, end: float) -> list[float]:
    s, e = float(start), float(end)
    if e < s:
        raise ValueError(f"Expected start ({s}) <= end ({e})")
    return [s] + [float(p) for p in grid if s + 1e-6 < float(p) < e - 1e-6] + [e]


def preview_progress_subgrid(grid: Sequence[float], start: float, steps: int = 4) -> list[float]:
    k = max(2, int(steps))
    tail = segment_progress_grid(grid, start, 1.0)
    if len(tail) < 3:
        return [float(start) + (1.0 - float(start)) * i / k for i in range(k + 1)]
    eff_k = min(k, len(tail) - 1)
    indices = sorted({round(i * (len(tail) - 1) / eff_k) for i in range(eff_k + 1)})
    return [float(tail[i]) for i in indices]


class GeneratorAdapter(ABC):
    """Unified adapter contract for any frozen flow-matching T2I generator."""

    corrector: StateCorrector | None = None

    @property
    @abstractmethod
    def geometry(self) -> LinearFlowGeometry:
        """Exact linear flow path & parameterization for this generator."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Primary execution device."""

    @abstractmethod
    def progress_grid(self) -> list[float]:
        """Monotonically increasing native sampling progress nodes in [0.0, 1.0]."""

    @abstractmethod
    def sample_initial_noise(self, *, seed: int, batch_size: int = 1) -> torch.Tensor:
        """Sample initial noise tensor z_0 at p=0 matching this generator's native shape/scale."""

    @abstractmethod
    def encode_prompt(self, text: str | None) -> ConditionBundle:
        """Encode native T2I text prompt (or unconditional empty prompt when text is None)."""

    @abstractmethod
    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor:
        """Encode RGB image tensor in [-1, 1] into the generator's clean flow space (pixel or VAE latent)."""

    @abstractmethod
    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image:
        """Decode a clean tensor in flow space into a displayable RGB PIL image."""

    @abstractmethod
    def predict_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        cond: ConditionBundle,
        *,
        use_corrector: bool = False,
    ) -> torch.Tensor:
        """Predict clean data/latent x_0 from noisy state z at progress p."""

    def encode_edit_condition(
        self,
        *,
        feedback: str,
        original_prompt: str | None = None,
        target_rewrite: str | None = None,
        condition_mode: ConditionMode = "native_prompt_plus_feedback",
    ) -> ConditionBundle:
        """Build feedback conditioning according to ``condition_mode``."""
        fb = " ".join(str(feedback or "").split()).strip()
        if not fb:
            return self.encode_prompt(None)
        if condition_mode == "feedback_only" or not original_prompt:
            text = f"Edit request: {fb}"
        elif condition_mode == "native_prompt_plus_feedback":
            orig = " ".join(str(original_prompt).split()).strip()
            text = f"Prompt: {orig}\nEdit request: {fb}"
        else:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        bundle = self.encode_prompt(text)
        bundle.mode = condition_mode
        return bundle

    @torch.no_grad()
    def _native_guided_clean_and_v(
        self,
        z: torch.Tensor,
        p: float,
        cond: ConditionBundle,
        uncond: ConditionBundle | None,
        cfg_scale: float,
        cfg_interval: tuple[float, float] = (0.0, 0.80),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pos = self.predict_clean(z, p, cond, use_corrector=False)
        v_pos = self.geometry.from_clean_to_v_clean(z, p, x_pos)
        lo, hi = float(cfg_interval[0]), float(cfg_interval[1])
        if uncond is None or float(cfg_scale) <= 1.0 + 1e-6 or not (lo <= p <= hi):
            return x_pos, v_pos
        x_neg = self.predict_clean(z, p, uncond, use_corrector=False)
        v_neg = self.geometry.from_clean_to_v_clean(z, p, x_neg)
        v_cfg = v_neg + float(cfg_scale) * (v_pos - v_neg)
        return x_pos, v_cfg

    @torch.no_grad()
    def sample_until(
        self,
        z_init: torch.Tensor,
        target_p: float,
        cond: ConditionBundle,
        uncond: ConditionBundle | None,
        cfg_scale: float,
        *,
        start_p: float = 0.0,
        cfg_interval: tuple[float, float] = (0.0, 0.80),
        return_clean_at_end: bool = True,
    ) -> tuple[LatentState, torch.Tensor | None]:
        """Integrate frozen native flow from ``start_p`` to ``target_p``."""
        times = segment_progress_grid(self.progress_grid(), start_p, target_p)
        z = z_init.detach().float().clone()
        for left, right in zip(times[:-1], times[1:]):
            _, v = self._native_guided_clean_and_v(z, left, cond, uncond, cfg_scale, cfg_interval)
            z = z + (right - left) * v
        cached_clean = (
            self.predict_clean(z, target_p, cond, use_corrector=False)
            if return_clean_at_end
            else None
        )
        return LatentState(tensor=z, progress=float(target_p)), cached_clean

    @torch.no_grad()
    def lookahead_preview(
        self,
        state: LatentState,
        cond: ConditionBundle,
        uncond: ConditionBundle | None,
        cfg_scale: float,
        *,
        steps: int = 4,
        cfg_interval: tuple[float, float] = (0.0, 0.80),
    ) -> Image.Image:
        """Multi-step Heun+Euler lookahead preview on a detached copy without mutating ``state``."""
        sub = preview_progress_subgrid(self.progress_grid(), state.progress, steps=steps)
        z = state.tensor.detach().float().clone()

        # First interval uses 2nd-order Heun predictor-corrector for sharp object emergence
        dt0 = sub[1] - sub[0]
        _, v0 = self._native_guided_clean_and_v(z, sub[0], cond, uncond, cfg_scale, cfg_interval)
        proposal = z + dt0 * v0
        _, v1 = self._native_guided_clean_and_v(proposal, sub[1], cond, uncond, cfg_scale, cfg_interval)
        z = z + 0.5 * dt0 * (v0 + v1)

        for left, right in zip(sub[1:-1], sub[2:]):
            _, v = self._native_guided_clean_and_v(z, left, cond, uncond, cfg_scale, cfg_interval)
            z = z + (right - left) * v
        return self.decode_preview(z)

    @torch.no_grad()
    def continue_native(
        self,
        state: LatentState,
        cond: ConditionBundle,
        uncond: ConditionBundle | None,
        cfg_scale: float,
        *,
        cfg_interval: tuple[float, float] = (0.0, 0.80),
    ) -> torch.Tensor:
        """Hand state back to frozen native generator to finish sampling to p=1.0."""
        final, _ = self.sample_until(
            state.tensor, 1.0, cond, uncond, cfg_scale, start_p=state.progress,
            cfg_interval=cfg_interval, return_clean_at_end=False,
        )
        return final.tensor
