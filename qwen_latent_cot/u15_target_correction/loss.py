"""Clean-target regression loss over valid (non-letterbox) canvas pixels."""


def clean_target_mse(predicted_clean, target_clean, valid=None):
    """Full-frequency clean-image MSE strictly inside valid canvas pixels."""
    error = (predicted_clean.float() - target_clean.float()).square()
    if valid is None:
        return error.mean()
    weight = valid.float()
    return (error * weight).sum() / weight.sum().clamp_min(1.0) / error.shape[1]
