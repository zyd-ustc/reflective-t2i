"""State flow matching and the evaluation-only image metrics."""

import torch


def _masked_mean(error, valid):
    weight = valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1) / error.shape[1]


def state_flow_matching(predicted_speed, target_speed, time, valid=None):
    """mean_valid [((u - (b-a)) / t)^2] with a strict interior time."""
    time = float(time)
    if not time > 0:
        raise ValueError("State flow matching needs a positive native time")
    error = ((predicted_speed - target_speed) / time).square()
    if valid is None:
        return error.mean()
    return _masked_mean(error, valid)


def endpoint_flow_matching(predicted_endpoint, target_endpoint, time, valid=None):
    """Remaining-endpoint loss mean_valid [((b_hat_k - b_t) / t)^2].

    At q_k = q_0 (s_0 = 0) or on the ideal linear interpolation path, this
    degenerates algebraically to ``(1 - s)^2 * ((u - (b_t - a_t)) / t)^2``.
    """
    return state_flow_matching(predicted_endpoint, target_endpoint, time, valid=valid)


def tail_flow_matching(predicted_clean, state, target_clean, tau, t_eps, valid=None):
    """Velocity-space tail loss using the native clamped x-to-v conversion.

    On z_tau = (1 - alpha) q_K + alpha x_b with tau = t_0 + (1 - t_0) alpha,
    (x_b - z_tau) / (1 - tau) == (x_b - q_K) / (1 - t_0). Clamping the
    denominator by ``t_eps`` matches ``_t2i_predict_v`` so the minimizer is
    strictly ``predicted_clean == target_clean`` up to ``tau -> 1``.
    """
    denom = max(1.0 - float(tau), float(t_eps))
    predicted_velocity = (predicted_clean.float() - state.float()) / denom
    target_velocity = (target_clean.float() - state.float()) / denom
    error = (predicted_velocity - target_velocity).square()
    if valid is None:
        return error.mean()
    return _masked_mean(error, valid)
