"""Strict isochronic manifold editing and UND-guided closed-loop T2I inference."""

import time as _time
import torch

from . import EDITOR_INTERFACE
from .diagnosis import review_reflection
from .field import (
    NativeT2I,
    euler,
    isochronic_manifold_transition,
    phase_grid,
    preview_image,
    segment_times,
)
from .images import load_canvas, load_pair


def _noised_source(rt, cfg, image_path, seed):
    image, _, _ = load_canvas(image_path, rt.width, rt.height, rt.device)
    t0 = float(cfg["read_time"])
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn(image.shape, device=rt.device, generator=generator) * rt.noise_scale()
    return image, t0 * image + (1.0 - t0) * noise, t0


@torch.no_grad()
def edit_pair(
    rt,
    editor,
    cfg,
    image_path,
    instruction,
    seed,
    load_trained=None,
    trained_editor=None,
    *,
    edit_cfg_scale=None,
    tail_prompt=None,
):
    """Edit one shared noised source exclusively via strict isochronic manifold transition + NativeT2I tail."""
    if load_trained and trained_editor is not None:
        raise ValueError("Pass either load_trained or trained_editor, not both")
    cfg_scale = float(cfg.get("edit_cfg_scale", 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    lookahead = int(cfg["lookahead_steps"])
    image, state, t0 = _noised_source(rt, cfg, image_path, seed)
    grid = phase_grid(rt, cfg)
    native = NativeT2I(rt, tail_prompt, cfg)
    baseline = euler(state.clone(), segment_times(grid, t0, 1.0), native)

    native_edited = None
    loaded_step = None
    active_editor = editor
    if load_trained or trained_editor is not None:
        native_edited, _, _, _ = isochronic_manifold_transition(
            editor, state.clone(), t0, instruction, grid, edit_cfg_scale=cfg_scale
        )
        if load_trained:
            loaded = load_trained()
            loaded_step = loaded.get("step") if isinstance(loaded, dict) else None
        else:
            active_editor = trained_editor

    began = _time.perf_counter()
    edited_state, clean_pred, _, tr_info = isochronic_manifold_transition(
        active_editor, state.clone(), t0, instruction, grid, edit_cfg_scale=cfg_scale
    )
    tr_seconds = _time.perf_counter() - began
    result = euler(edited_state.clone(), segment_times(grid, t0, 1.0), native)
    oob_rate = float(((result < -1.0) | (result > 1.0)).float().mean().item())

    images = {
        "source": image,
        "baseline": baseline,
        "clean_estimate": clean_pred,
        "edited_preview": preview_image(edited_state, t0, grid, native, steps=lookahead),
        "result": result,
    }
    states = {"result": edited_state}
    if native_edited is not None:
        images["native_edited_preview"] = preview_image(native_edited, t0, grid, native, steps=lookahead)
        images["native_result"] = euler(native_edited.clone(), segment_times(grid, t0, 1.0), native)
        states["native_result"] = native_edited

    return images, states, dict(
        editor_interface=EDITOR_INTERFACE,
        mode="edit",
        instruction=instruction,
        seed=seed,
        input_time=t0,
        stop_time=tr_info["stop_time"],
        output_time=tr_info["output_time"],
        times=tr_info["times"],
        edit_cfg_scale=cfg_scale,
        lookahead_steps=lookahead,
        anchor_nfe=tr_info["anchor_nfe"],
        correction_nfe=tr_info["correction_nfe"],
        transition_seconds=tr_seconds,
        raw_oob_rate=oob_rate,
        continuation="frozen_native_t2i_g1",
        loaded_checkpoint_step=loaded_step,
    )


@torch.no_grad()
def isochronic_edit_pair_row(rt, editor, cfg, row, seed, *, edit_cfg_scale=None):
    """Evaluate strict isochronic manifold transition + frozen NativeT2I tail on a paired validation row."""
    source, target, valid, _ = load_pair(row, rt.width, rt.height, rt.device)
    t0 = float(cfg["read_time"])
    cfg_scale = float(cfg.get("edit_cfg_scale", 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn(source.shape, device=rt.device, generator=generator) * rt.noise_scale()
    anchor = t0 * source + (1.0 - t0) * noise
    grid = phase_grid(rt, cfg)
    instruction = row.get("edit_instruction") or row.get("edit_request")
    tail_prompt = (row.get("tail_condition") or row.get("target_caption") or "").strip() or None
    native = NativeT2I(rt, tail_prompt, cfg)

    began = _time.perf_counter()
    z_iso, clean_pred, _, tr_info = isochronic_manifold_transition(
        editor, anchor, t0, instruction, grid, edit_cfg_scale=cfg_scale
    )
    final = euler(z_iso, segment_times(grid, t0, 1.0), native)
    elapsed = _time.perf_counter() - began
    oob_rate = float(((final < -1.0) | (final > 1.0)).float().mean().item())

    return (
        {"source": source, "target": target, "clean_estimate": clean_pred, "result": final},
        z_iso,
        dict(
            editor_interface=EDITOR_INTERFACE,
            mode="pair_eval",
            instruction=instruction,
            seed=seed,
            valid=valid,
            input_time=tr_info["input_time"],
            stop_time=tr_info["stop_time"],
            output_time=tr_info["output_time"],
            times=tr_info["times"],
            edit_cfg_scale=cfg_scale,
            anchor_nfe=tr_info["anchor_nfe"],
            correction_nfe=tr_info["correction_nfe"],
            total_seconds=elapsed,
            raw_oob_rate=oob_rate,
            continuation="frozen_native_t2i_g1",
        ),
    )


@torch.no_grad()
def edit_image(rt, editor, cfg, image_path, instruction, seed, *, edit_cfg_scale=None):
    images, states, record = edit_pair(
        rt, editor, cfg, image_path, instruction, seed, edit_cfg_scale=edit_cfg_scale
    )
    return images, states["result"], record


@torch.no_grad()
def t2i_feedback(rt, editor, cfg, prompt, seed, *, edit_cfg_scale=None):
    """Closed-loop T2I using multi-step lookahead preview, gated UND reflection, and strict isochronic manifold jump."""
    cfg_scale = float(cfg.get("edit_cfg_scale", 1.0) if edit_cfg_scale is None else edit_cfg_scale)
    lookahead = int(cfg["lookahead_steps"])
    grid = phase_grid(rt, cfg)
    t0 = float(cfg["read_time"])
    native = NativeT2I(rt, prompt, cfg)

    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
    state = euler(noise, segment_times(grid, 0.0, t0), native)
    preview = preview_image(state, t0, grid, native, steps=lookahead)
    review = review_reflection(rt, preview, prompt, cfg)["reflection"]
    if review["outcome"] in ("failed_parse", "failed_unconverted"):
        raise ValueError("UND reflection could not produce a decision: " + review["raw"])

    images = {"preview": preview}
    edited_state = None
    tr_seconds = 0.0
    anchor_nfe = 0
    correction_nfe = 0
    tr_times = []
    if review["outcome"] == "edit":
        began = _time.perf_counter()
        edited_state, clean_pred, _, tr_info = isochronic_manifold_transition(
            editor, state, t0, review["instruction"], grid, edit_cfg_scale=cfg_scale
        )
        tr_seconds = _time.perf_counter() - began
        anchor_nfe = tr_info["anchor_nfe"]
        correction_nfe = tr_info["correction_nfe"]
        tr_times = tr_info["times"]
        images["clean_estimate"] = clean_pred
        images["edited_preview"] = preview_image(edited_state, t0, grid, native, steps=lookahead)

    tail_start = edited_state if edited_state is not None else state
    result = euler(tail_start, segment_times(grid, t0, 1.0), native)
    images["result"] = result
    oob_rate = float(((result < -1.0) | (result > 1.0)).float().mean().item())

    return images, edited_state, dict(
        editor_interface=EDITOR_INTERFACE,
        mode="t2i",
        prompt=prompt,
        seed=seed,
        input_time=t0,
        stop_time=float(cfg["stop_time"]),
        output_time=t0,
        times=tr_times,
        edit_cfg_scale=cfg_scale,
        lookahead_steps=lookahead,
        anchor_nfe=anchor_nfe,
        correction_nfe=correction_nfe,
        transition_seconds=tr_seconds,
        raw_oob_rate=oob_rate,
        continuation="frozen_native_t2i_g1",
        reflection=review,
    )
