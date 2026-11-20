"""Single-record benchmark generation using strict isochronic manifold transition."""

import torch
from PIL import Image

from qwen_latent_cot.u15_target_correction.diagnosis import review_reflection
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I,
    euler,
    isochronic_manifold_transition,
    phase_grid,
    preview_image,
    segment_times,
)


def _transition(editor, state, time, instruction, grid):
    return isochronic_manifold_transition(editor, state, time, instruction, grid)[0]


@torch.no_grad()
def t2i_pair(rt, editor, cfg, prompt, seed, *, load_trained=None):
    grid, t = phase_grid(rt, cfg), float(cfg["read_time"])
    lookahead = int(cfg.get("lookahead_steps", 4))
    native = NativeT2I(rt, prompt, cfg)
    generator = torch.Generator(device=rt.device).manual_seed(int(seed))
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * rt.noise_scale()
    state = euler(noise, segment_times(grid, 0, t), native)
    preview = preview_image(state, t, grid, native, steps=lookahead)
    review = review_reflection(rt, preview, prompt, cfg)["reflection"]
    if review["outcome"] in ("failed_parse", "failed_unconverted"):
        raise ValueError("UND reflection could not produce a decision: " + review["raw"])
    baseline = euler(state.clone(), segment_times(grid, t, 1), native)
    images = {"preview": preview, "baseline": baseline}
    if review["outcome"] == "edit":
        if load_trained:
            initial = _transition(editor, state, t, review["instruction"], grid)
            images["native_result"] = euler(initial, segment_times(grid, t, 1), native)
            load_trained()
        edited = _transition(editor, state, t, review["instruction"], grid)
        images["edited_preview"] = preview_image(edited, t, grid, native, steps=lookahead)
        images["result"] = euler(edited, segment_times(grid, t, 1), native)
    else:
        if load_trained:
            images["native_result"] = baseline
            load_trained()
        images["result"] = baseline
    return images, {"reflection": review, "read_time": t}


def letterbox(source, width, height):
    source = source.convert("RGB")
    scale = min(width / source.width, height / source.height)
    size = (round(source.width * scale), round(source.height * scale))
    offset = ((width - size[0]) // 2, (height - size[1]) // 2)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(source.resize(size, Image.Resampling.LANCZOS), offset)
    return canvas
