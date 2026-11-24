"""Strict isochronic manifold transition, subgrid lookahead preview, and frozen native T2I."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import GEOMETRY_PROTOCOL, KIND, PROTOCOL_VERSION


def protocol(rt, cfg):
    """Frozen discretisation fingerprint bound to cached manifests."""
    nodes = [float(cfg["read_time"]), float(cfg["stop_time"])]
    return dict(
        phase_grid_times=sorted(set(nodes)),
        kind=KIND,
        protocol_version=PROTOCOL_VERSION,
        geometry=GEOMETRY_PROTOCOL,
        native=rt.identity,
        width=rt.width,
        height=rt.height,
        noise_scale=rt.noise_scale(),
        times=[float(t) for t in rt.times()],
        sampler="euler",
        cfg_scale=float(cfg["cfg_scale"]),
        image_cfg_scale=float(cfg["image_cfg_scale"]),
        cfg_norm=cfg["cfg_norm"],
        cfg_interval=[float(t) for t in cfg["cfg_interval"]],
        t_eps=float(cfg["t_eps"]),
        read_time=float(cfg["read_time"]),
        stop_time=float(cfg["stop_time"]),
        lookahead_steps=int(cfg["lookahead_steps"]),
        condition_protocol=cfg["condition_protocol"],
        preview_protocol="subgrid_heun_euler_v2",
        tail_solver="euler",
    )


def phase_grid(rt, cfg):
    """Deployment ODE solver grid: native shifted grid plus read_time and stop_time."""
    values = [float(t) for t in rt.times()]
    extras = [float(cfg["read_time"]), float(cfg["stop_time"])]
    extras += [float(t) for t in cfg.get("phase_grid_times", ())]
    for value in extras:
        values = [t for t in values if abs(t - value) > 1e-7] + [value]
    return sorted(values)


def segment_times(grid, start, end):
    start, end = float(start), float(end)
    if end < start:
        raise ValueError("segment_times expects start <= end")
    return [start] + [float(t) for t in grid if start + 1e-7 < t < end - 1e-7] + [end]


def time_key(value):
    return f"{float(value):.6f}"


def euler(z, times, velocity):
    """Discrete Euler integration along ``times``."""
    z = z.float()
    for left, right in zip(times[:-1], times[1:]):
        step = z.new_tensor(float(right) - float(left))
        moment = z.new_tensor(float(left))

        def call(state, moment=moment):
            return velocity(state, moment)

        if torch.is_grad_enabled():
            speed = checkpoint(call, z, use_reentrant=False)
        else:
            speed = call(z)
        z = z + step * speed
    return z


def preview_subgrid(grid, start_time, steps=4):
    """Select ``steps`` evenly spaced sub-intervals on ``[start_time, 1.0]``."""
    k = int(steps)
    if k < 2:
        raise ValueError(f"preview_image requires steps >= 2, got {steps}")
    tail = segment_times(grid, start_time, 1.0)
    if len(tail) < 3:
        raise ValueError("Native grid has no interior preview node after start_time")
    eff_k = min(k, len(tail) - 1)
    indices = sorted({round(i * (len(tail) - 1) / eff_k) for i in range(eff_k + 1)})
    sub = [float(tail[i]) for i in indices]
    if len(sub) < 3 or not float(start_time) < sub[1] < 1.0:
        raise ValueError("Preview subgrid failed to resolve an interior node")
    return sub


@torch.no_grad()
def preview_image(state, start_time, grid, velocity, *, steps=4):
    """Multi-step subgrid lookahead on a detached copy (Heun on first interval + Euler).

    Increasing ``steps`` from 2 -> 4 resolves the [0.36, 0.75] object-emergence
    band so secondary and small objects crystallize before UND inspection,
    without mutating ``state``.
    """
    original = state.detach().clone()
    sub = preview_subgrid(grid, start_time, steps=steps)
    z = original.clone().float()

    # Heun predictor-corrector on the highest-curvature initial interval [t0, sub[1]]
    t_left = z.new_tensor(sub[0])
    t_next = z.new_tensor(sub[1])
    dt0 = float(sub[1] - sub[0])
    v0 = velocity(z, t_left)
    proposal = z + dt0 * v0
    z = z + 0.5 * dt0 * (v0 + velocity(proposal, t_next))

    # Remaining intervals use standard Euler steps along the shifted subgrid
    for left, right in zip(sub[1:-1], sub[2:]):
        z = z + float(right - left) * velocity(z, z.new_tensor(float(left)))

    if not torch.equal(state.detach(), original):
        raise RuntimeError("Preview mutated the main trajectory")
    return z


class NativeT2I:
    """Frozen native text-to-image CFG velocity field (never touches editor LoRA)."""

    def __init__(self, rt, raw, cfg):
        self.rt, self.cfg = rt, cfg
        self.positive = rt.prefix(raw)
        self.negative = rt.prefix(None) if float(cfg["cfg_scale"]) > 1 else None
        self.calls = 0

    def __call__(self, z, t):
        self.calls += 1
        positive = self.rt.predict(z, t, self.positive).float()
        lo, hi = [float(v) for v in self.cfg["cfg_interval"]]
        if self.negative is None or not lo <= float(t) <= hi:
            return positive
        self.calls += 1
        negative = self.rt.predict(z, t, self.negative).float()
        return self.rt.guided(positive, negative, float(self.cfg["cfg_scale"])).float()


@torch.no_grad()
def compute_self_emergent_edit_mask(
    clean_target,
    clean_source,
    *,
    low_thresh=0.10,
    high_thresh=0.32,
    pool_size=16,
    blur_sigma=1.0,
):
    """Derive a smooth, bbox-free spatial edit mask in [0, 1] from (x_hat_T - x_hat_S).

    Operates at patch-pooled resolution so high-frequency sharpness differences between
    ``t0`` and ``stop_time`` do not trigger false mask activations on untouched objects.
    """
    t_f = clean_target.detach().float()
    s_f = clean_source.detach().float()
    _, _, h, w = t_f.shape
    k = max(1, min(int(pool_size), h, w))
    if k > 1 and h % k == 0 and w % k == 0:
        t_low = F.avg_pool2d(t_f, kernel_size=k, stride=k)
        s_low = F.avg_pool2d(s_f, kernel_size=k, stride=k)
    else:
        t_low, s_low = t_f, s_f

    diff = (t_low - s_low).square().mean(dim=1, keepdim=True).sqrt()
    lo, hi = float(low_thresh), float(high_thresh)
    denom = max(hi - lo, 1e-6)
    mask_low = ((diff - lo) / denom).clamp(0.0, 1.0)

    sigma = float(blur_sigma)
    if sigma > 0.0 and mask_low.shape[-2] >= 3 and mask_low.shape[-1] >= 3:
        radius = max(1, int(round(2.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=mask_low.device, dtype=torch.float32)
        kern1d = torch.exp(-0.5 * (coords / sigma).square())
        kern1d = kern1d / kern1d.sum()
        kh = kern1d.view(1, 1, 1, -1)
        kv = kern1d.view(1, 1, -1, 1)
        mask_low = F.pad(mask_low, (radius, radius, 0, 0), mode="replicate")
        mask_low = F.conv2d(mask_low, kh)
        mask_low = F.pad(mask_low, (0, 0, radius, radius), mode="replicate")
        mask_low = F.conv2d(mask_low, kv)
        mask_low = mask_low.clamp(0.0, 1.0)

    if mask_low.shape[-2:] != (h, w):
        mask = F.interpolate(mask_low, size=(h, w), mode="bilinear", align_corners=False)
    else:
        mask = mask_low
    mask = mask.clamp(0.0, 1.0)

    flat = diff.reshape(-1)
    stats = {
        "diff_p50": float(torch.quantile(flat, 0.50).item()),
        "diff_p90": float(torch.quantile(flat, 0.90).item()),
        "diff_p95": float(torch.quantile(flat, 0.95).item()),
        "diff_max": float(flat.max().item()),
        "mask_mean": float(mask.mean().item()),
        "mask_max": float(mask.max().item()),
    }
    return mask, stats


@torch.no_grad()
def project_clean_to_native_manifold(
    clean_pred,
    eps_anchor,
    output_time,
    *,
    sigma=1.0,
    noise_refresh=0.0,
    noise_refresh_map=None,
    preserve_state=None,
    edit_mask=None,
    generator=None,
):
    """Project a clean target estimate onto the native noisy manifold at ``output_time``."""
    t_out = float(output_time)
    if not 0.0 < t_out < 1.0:
        raise ValueError(f"output_time must lie in (0, 1), got {output_time}")
    eps_eff = eps_anchor.float()
    if noise_refresh_map is not None:
        eta_map = noise_refresh_map.float().clamp(0.0, 1.0)
        if float(eta_map.max().item()) > 0.0:
            xi = torch.randn(
                eps_eff.shape,
                device=eps_eff.device,
                dtype=torch.float32,
                generator=generator,
            ) * float(sigma)
            eps_eff = (1.0 - eta_map.square()).clamp_min(0.0).sqrt() * eps_eff + eta_map * xi
    else:
        eta = float(noise_refresh)
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"noise_refresh must lie in [0, 1], got {noise_refresh}")
        if eta > 0.0:
            xi = torch.randn(
                eps_eff.shape,
                device=eps_eff.device,
                dtype=torch.float32,
                generator=generator,
            ) * float(sigma)
            eps_eff = ((1.0 - eta * eta) ** 0.5) * eps_eff + eta * xi
    z_reproj = t_out * clean_pred.float() + (1.0 - t_out) * eps_eff
    if preserve_state is not None and edit_mask is not None:
        m = edit_mask.float().clamp(0.0, 1.0)
        return (1.0 - m) * preserve_state.float() + m * z_reproj
    return z_reproj


@torch.no_grad()
def isochronic_manifold_transition(
    editor,
    state,
    time,
    feedback,
    grid,
    *,
    stop_time=None,
    output_time=None,
    edit_cfg_scale=None,
    noise_refresh=None,
    anchor_prefix=None,
    empty_prefix=None,
    preserve_state=None,
    soft_mask=False,
    mask_low=0.10,
    mask_high=0.32,
    mask_pool=16,
    mask_blur_sigma=1.0,
    masked_noise_refresh=0.0,
    generator=None,
):
    """Strict isochronic manifold jump: z_{t0}^{(S)} -> z_hat_{t_out}^{(T)}.

    1. Extract frozen native clean anchor x_hat_S = H_0(z_{t0}, t0, anchor_prefix) and
       structural noise anchor eps_anchor = (z_{t0} - t0 * x_hat_S) / (1 - t0).
    2. Roll out adapted short-horizon semantic flow q_tau on [t0, stop_time]
       and read out clean target estimate x_hat_T = H_theta(q_{stop_time}, stop_time, feedback).
    3. Reproject onto the native noisy manifold at ``output_time`` (default ``t0``), optionally
       using a bbox-free self-emergent edit mask for non-edit preservation and localized noise refresh.
    """
    t0 = float(time)
    expected = float(editor.config["read_time"])
    if abs(t0 - expected) > 1e-6:
        raise ValueError(
            f"isochronic_manifold_transition requires time == read_time ({expected}), got {time}"
        )
    t_stop = float(editor.config["stop_time"] if stop_time is None else stop_time)
    if not t0 < t_stop < 1.0:
        raise ValueError(f"stop_time must satisfy read_time ({t0}) < stop_time < 1.0, got {t_stop}")
    t_out = t0 if output_time is None else float(output_time)
    if not t0 - 1e-6 <= t_out < 1.0:
        raise ValueError(f"output_time must satisfy read_time ({t0}) <= output_time < 1.0, got {t_out}")

    cfg_scale = float(editor.config.get("edit_cfg_scale", 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    if cfg_scale < 1.0:
        raise ValueError(f"edit_cfg_scale must be >= 1.0, got {cfg_scale}")
    eta = float(editor.config.get("noise_refresh", 0.0) if noise_refresh is None else noise_refresh)

    empty_pref = editor.empty_prefix() if empty_prefix is None else empty_prefix
    anchor_pref = empty_pref if anchor_prefix is None else anchor_prefix
    x_source_est = editor.native_clean(state.detach(), t0, anchor_pref)
    eps_anchor = (state.float() - t0 * x_source_est.float()) / (1.0 - t0)

    cond_pref = editor.feedback_prefix(feedback)
    times = segment_times(grid, t0, t_stop)
    calls_before = editor.clean_calls
    q_stop = euler(
        state.detach().clone(),
        times,
        lambda z, tau: editor.predict_velocity(
            z, tau, cond_pref, empty_prefix=empty_pref, cfg_scale=cfg_scale
        ),
    )
    clean_pred = editor.predict_guided_clean(
        q_stop, t_stop, cond_pref, empty_prefix=empty_pref, cfg_scale=cfg_scale
    )

    edit_mask = None
    mask_stats = {}
    eta_map = None
    if soft_mask:
        edit_mask, mask_stats = compute_self_emergent_edit_mask(
            clean_pred,
            x_source_est,
            low_thresh=mask_low,
            high_thresh=mask_high,
            pool_size=mask_pool,
            blur_sigma=mask_blur_sigma,
        )
        m_eta = float(masked_noise_refresh)
        if m_eta > 0.0:
            eta_map = (m_eta * edit_mask).clamp(0.0, 1.0)
        if preserve_state is None and abs(t_out - t0) <= 1e-6:
            preserve_state = state.detach()

    z_iso = project_clean_to_native_manifold(
        clean_pred,
        eps_anchor,
        t_out,
        sigma=editor.rt.noise_scale(),
        noise_refresh=eta,
        noise_refresh_map=eta_map,
        preserve_state=preserve_state if soft_mask else None,
        edit_mask=edit_mask,
        generator=generator,
    )
    info = {
        "input_time": t0,
        "stop_time": t_stop,
        "output_time": t_out,
        "times": [float(t) for t in times],
        "edit_cfg_scale": cfg_scale,
        "noise_refresh": eta,
        "soft_mask": bool(soft_mask),
        "masked_noise_refresh": float(masked_noise_refresh),
        "anchor_nfe": 1,
        "correction_nfe": editor.clean_calls - calls_before,
    }
    info.update(mask_stats)
    return z_iso, clean_pred, eps_anchor, info
