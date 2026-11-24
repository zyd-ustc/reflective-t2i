"""Unified Linear Flow Matching geometry over normalized denoising progress p in [0, 1].

Convention across reflective_t2i_v1:
  - p = 0.0 : pure noise manifold (z_0 = noise_scale * eps)
  - p = 1.0 : clean data/latent manifold (z_1 = x_0)
  - z_p = p * x_0 + (1 - p) * eps_scaled
  - v_to_clean = dz_p / dp = x_0 - eps_scaled = (x_0 - z_p) / (1 - p)

Supported native generator prediction parameterizations:
  - "clean_x"    : network directly predicts x_0 (SenseNova-U1.5 fm_head)
  - "v_to_clean" : network predicts v = x_0 - eps (Z-Image S3-DiT)
  - "v_to_noise" : network predicts v = eps - x_0 (Diffusers SD3.5 / FLUX FlowMatch)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import PredictionType


def _expand_p(p: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    t = torch.as_tensor(p, device=ref.device, dtype=torch.float32)
    if t.ndim == 0:
        t = t.expand(ref.shape[0])
    while t.ndim < ref.ndim:
        t = t.unsqueeze(-1)
    return t


@dataclass(frozen=True)
class LinearFlowGeometry:
    """Exact algebraic conversions between z_p, x_0, eps, and native model predictions."""

    prediction_type: PredictionType = "v_to_clean"
    noise_scale: float = 1.0
    t_eps: float = 0.05

    def __post_init__(self) -> None:
        if self.prediction_type not in ("clean_x", "v_to_clean", "v_to_noise"):
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")
        if self.noise_scale <= 0.0:
            raise ValueError(f"noise_scale must be positive, got {self.noise_scale}")
        if not 0.0 < self.t_eps < 0.5:
            raise ValueError(f"t_eps must lie in (0, 0.5), got {self.t_eps}")

    def sample_noise(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        eps = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return (eps * float(self.noise_scale)).to(dtype=dtype)

    def interpolate(
        self,
        x_clean: torch.Tensor,
        eps_scaled: torch.Tensor,
        p: float | torch.Tensor,
    ) -> torch.Tensor:
        """Forward noisy state z_p = p * x_clean + (1 - p) * eps_scaled."""
        pt = _expand_p(p, x_clean)
        return pt * x_clean.float() + (1.0 - pt) * eps_scaled.float()

    def to_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        native_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convert any native prediction (clean_x / v_to_clean / v_to_noise) to clean x_0."""
        zf = z.float()
        pf = native_pred.float()
        pt = _expand_p(p, zf)
        if self.prediction_type == "clean_x":
            return pf
        if self.prediction_type == "v_to_clean":
            return zf + (1.0 - pt) * pf
        # v_to_noise: v = eps - x_0 => z_p - (1 - p) * v = x_0
        return zf - (1.0 - pt) * pf

    def to_v_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        native_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Convert any native prediction to v_to_clean = dz_p / dp = x_0 - eps_scaled."""
        zf = z.float()
        pf = native_pred.float()
        if self.prediction_type == "v_to_clean":
            return pf
        if self.prediction_type == "v_to_noise":
            return -pf
        pt = _expand_p(p, zf)
        denom = (1.0 - pt).clamp_min(self.t_eps)
        return (pf - zf) / denom

    def from_clean_to_v_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Convert clean estimate x_0 to flow velocity v_to_clean = (x_0 - z_p) / max(1 - p, t_eps)."""
        pt = _expand_p(p, z)
        denom = (1.0 - pt).clamp_min(self.t_eps)
        return (x_clean.float() - z.float()) / denom

    def extract_eps_anchor(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        x_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Invert z_p = p * x_clean + (1 - p) * eps_anchor for eps_anchor."""
        pt = _expand_p(p, z)
        denom = (1.0 - pt).clamp_min(self.t_eps)
        return (z.float() - pt * x_clean.float()) / denom

    def to_eps(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        native_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scaled noise estimate eps_scaled directly from native prediction."""
        zf = z.float()
        pf = native_pred.float()
        pt = _expand_p(p, zf)
        if self.prediction_type == "v_to_clean":
            return zf - pt * pf
        if self.prediction_type == "v_to_noise":
            return zf + pt * pf
        return self.extract_eps_anchor(zf, pt, pf)

    def compose_state(
        self,
        x_clean: torch.Tensor,
        eps_anchor: torch.Tensor,
        p_out: float | torch.Tensor,
        *,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Project clean estimate onto native noisy manifold at p_out with optional CPS noise refresh."""
        eta_f = float(eta)
        if not 0.0 <= eta_f <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {eta}")
        eps_eff = eps_anchor.float()
        if eta_f > 0.0:
            xi = self.sample_noise(
                tuple(eps_eff.shape),
                device=eps_eff.device,
                dtype=torch.float32,
                generator=generator,
            )
            eps_eff = math.sqrt(max(0.0, 1.0 - eta_f * eta_f)) * eps_eff + eta_f * xi
        return self.interpolate(x_clean, eps_eff, p_out)

    def cps_step(
        self,
        z_curr: torch.Tensor,
        p_curr: float | torch.Tensor,
        p_next: float | torch.Tensor,
        x_pred: torch.Tensor,
        *,
        eta: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Coefficients-Preserving Sampling step (from HY-SOAR) advancing p_curr -> p_next."""
        eps_est = self.extract_eps_anchor(z_curr, p_curr, x_pred)
        sin_term = math.sin(max(0.0, min(1.0, float(eta))) * math.pi * 0.5)
        return self.compose_state(x_pred, eps_est, p_next, eta=sin_term, generator=generator)

    @staticmethod
    def trust_region_clip(
        delta: torch.Tensor,
        reference: torch.Tensor,
        rho: float,
        *,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """CorrFlow-style per-sample RMS trust-region clamping on edit displacement delta."""
        rho_f = float(rho)
        if rho_f <= 0.0:
            return delta
        df = delta.float()
        rf = reference.float()
        dims = tuple(range(1, df.ndim))
        d_rms = df.square().mean(dim=dims, keepdim=True).sqrt()
        r_rms = rf.square().mean(dim=dims, keepdim=True).sqrt()
        scale = (rho_f * r_rms / (d_rms + eps)).clamp(max=1.0)
        return (df * scale).to(dtype=delta.dtype)
