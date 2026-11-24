"""Model-agnostic training losses, SOAR self-rollout bridge, dataset, and loop."""

from .losses import (
    LossConfig,
    ReflectiveCorrectionLoss,
    clean_target_mse,
    cosine_residual_loss,
    delta_weighted_clean_mse,
)
from .rollout import RolloutConfig, build_training_batch_step

__all__ = [
    "LossConfig",
    "ReflectiveCorrectionLoss",
    "RolloutConfig",
    "build_training_batch_step",
    "clean_target_mse",
    "cosine_residual_loss",
    "delta_weighted_clean_mse",
]
