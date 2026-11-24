"""Zero-bbox training losses for reflective state correction.

Combines:
  1. Self-emergent Delta-Weighted Clean MSE: upweights pixels/latents where ||T - S||_2
     is large (e.g., 3%-8% small added/counted entities) without any bbox annotations.
  2. CorrFlow-style Cosine Residual Direction Loss: 1 - cos(x_hat - S, T - S) with the
     zero-init `has_direction` guard preventing 1/eps gradient explosion on step 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _align_valid_mask(valid: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor:
    if valid is None:
        return torch.ones((ref.shape[0], 1, *ref.shape[2:]), device=ref.device, dtype=torch.float32)
    mask = valid.to(device=ref.device, dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != ref.shape[-2:]:
        mask = F.adaptive_avg_pool2d(mask, ref.shape[-2:])
    return mask.clamp(0.0, 1.0)


def clean_target_mse(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standard valid-region clean MSE (legacy baseline parity)."""
    mask = _align_valid_mask(valid, predicted_clean)
    sq = (predicted_clean.float() - target_clean.float()).square().mean(dim=1, keepdim=True)
    per_sample = (sq * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return per_sample.mean()


def delta_weighted_clean_mse(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    source_clean: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    delta_lambda: float = 4.0,
    max_weight: float = 8.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Zero-bbox spatially weighted clean MSE driven by self-emergent ||T - S||_2."""
    mask = _align_valid_mask(valid, predicted_clean)
    with torch.no_grad():
        delta_mag = (target_clean.float() - source_clean.float()).square().mean(dim=1, keepdim=True).sqrt()
        mean_delta = (delta_mag * mask).sum(dim=(1, 2, 3), keepdim=True) / mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)
        # If sample is an identity/source_dropout sample (T == S), mean_delta ~ 0 -> weight stays 1.0
        has_edit = (mean_delta > eps).to(dtype=delta_mag.dtype)
        rel_mag = (delta_mag / (mean_delta + eps)).clamp(0.0, float(max_weight))
        spatial_weight = 1.0 + float(delta_lambda) * has_edit * rel_mag
        eff_weight = spatial_weight * mask

    sq = (predicted_clean.float() - target_clean.float()).square().mean(dim=1, keepdim=True)
    per_sample = (sq * eff_weight).sum(dim=(1, 2, 3)) / eff_weight.sum(dim=(1, 2, 3)).clamp_min(1.0)
    loss = per_sample.mean()
    with torch.no_grad():
        w_ratio = float((eff_weight.sum() / mask.sum().clamp_min(1.0)).item())
    return loss, {"edit_weight_ratio": w_ratio}


def cosine_residual_loss(
    predicted_clean: torch.Tensor,
    target_clean: torch.Tensor,
    source_clean: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CorrFlow-inspired cosine direction loss on edit delta (x_hat - S) vs (T - S)."""
    mask = _align_valid_mask(valid, predicted_clean).sqrt()
    pred_delta = ((predicted_clean.float() - source_clean.float()) * mask).flatten(1)
    true_delta = ((target_clean.float() - source_clean.float()) * mask).flatten(1)

    pred_norm = torch.linalg.vector_norm(pred_delta, dim=1)
    true_norm = torch.linalg.vector_norm(true_delta, dim=1)

    # Guard against undefined cos(0, r) at step 1 or on identity-dropout rows (T == S)
    has_direction = (pred_norm > eps) & (true_norm > eps)
    safe_denom = (pred_norm * true_norm).clamp_min(eps)
    cos = torch.einsum("bd,bd->b", pred_delta, true_delta) / safe_denom
    direction_term = torch.where(has_direction, 1.0 - cos, torch.zeros_like(cos))
    active = has_direction.float().sum().clamp_min(1.0)
    loss = direction_term.sum() / active
    with torch.no_grad():
        mean_cos = float((torch.where(has_direction, cos, torch.zeros_like(cos)).sum() / active).item())
        active_frac = float(has_direction.float().mean().item())
    return loss, {"delta_cos": mean_cos, "cos_active_frac": active_frac}


@dataclass
class LossConfig:
    mode: str = "delta_weighted_clean_mse_plus_cosine"
    delta_weight_lambda: float = 1.0
    delta_weight_max: float = 3.0
    cosine_lambda: float = 0.02


class ReflectiveCorrectionLoss:
    """Configurable loss combining delta-weighted clean MSE and cosine edit-direction loss."""

    def __init__(self, config: LossConfig | None = None):
        self.config = config or LossConfig()

    def __call__(
        self,
        predicted_clean: torch.Tensor,
        target_clean: torch.Tensor,
        source_clean: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mode = self.config.mode
        if mode == "clean_target_mse":
            loss = clean_target_mse(predicted_clean, target_clean, valid)
            return loss, {"loss_clean": float(loss.detach().item())}

        w_loss, w_stats = delta_weighted_clean_mse(
            predicted_clean,
            target_clean,
            source_clean,
            valid,
            delta_lambda=self.config.delta_weight_lambda,
            max_weight=self.config.delta_weight_max,
        )
        metrics = {"loss_wclean": float(w_loss.detach().item()), **w_stats}
        if mode == "delta_weighted_clean_mse" or self.config.cosine_lambda <= 0.0:
            return w_loss, metrics

        c_loss, c_stats = cosine_residual_loss(predicted_clean, target_clean, source_clean, valid)
        total = w_loss + float(self.config.cosine_lambda) * c_loss
        metrics.update({"loss_cos": float(c_loss.detach().item()), **c_stats})
        return total, metrics
