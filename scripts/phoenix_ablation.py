"""Why did the linear album add a phoenix while the closed loop cannot?

The album walked the training route on a real photograph: a clean source image,
mixed with the way ``train._linear_fields`` mixes it, driven by a 28-word
instruction that names where the new object goes. The closed loop walks a T2I
trajectory and hands over a shorter instruction. Four cells separate the two:

                album instruction      loop instruction
``linear``      A                      C        (training route, real photo)
``t2i``         B                      D        (deployment route, model pixels)

The instruction pair is (A, B): the album wording and the wording the loop
wrote for the same scene. The start pair is (A, C): the real source mixed the
training way, versus the state the model's own trajectory holds at the read
time.
"""

import argparse
import json
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, edit_state, euler, phase_grid, preview_image, segment_times,
)
from qwen_latent_cot.u15_target_correction.images import load_canvas, save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor

ALBUM_INSTRUCTION = (
    "Add 1 phoenix to the image so that it contains exactly 2 phoenixes instead of the "
    "current 1, placing the new ones in plausible empty positions; keep everything else unchanged."
)
LOOP_INSTRUCTION = (
    "add one phoenix so that 1 becomes 2, placing it in the empty sky to the right of the "
    "existing phoenix, keeping the existing phoenix unchanged"
)
PARENT = (
    "A majestic, fiery phoenix with vibrant orange and red plumage, expansive wings, and a "
    "flowing tail soars gracefully through a dramatic sky filled with clouds illuminated by "
    "the warm glow of a setting or rising sun."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--album-source", required=True,
                        help="The clean source image the album case started from")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20285973)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, set_device(args.device), False)
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))
    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    steps = int(cfg["edit_steps"])
    sigma = rt.noise_scale()
    tail_times = segment_times(grid, read, 1.0)
    args.out.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=rt.device).manual_seed(args.seed)
    noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                        generator=generator) * sigma

    # The album route: the real photograph, mixed the way training mixes it.
    photo, _, _ = load_canvas(Path(args.album_source), rt.width, rt.height, rt.device)
    linear_state = read * photo + (1 - read) * noise
    save_image(args.out / "source.png", photo)
    save_image(args.out / "linear_state.png", linear_state)

    # The deployment route: the model's own trajectory to the same read time.
    native = NativeT2I(rt, PARENT, cfg)
    t2i_state = euler(noise.float(), segment_times(grid, 0.0, read), native)
    save_image(args.out / "t2i_state.png", t2i_state)
    for tag in ("linear", "t2i"):
        state = linear_state if tag == "linear" else t2i_state
        save_image(args.out / f"{tag}_preview.png", preview_image(state, read, grid,
                                                                 NativeT2I(rt, PARENT, cfg)))

    cells = [
        ("A_linear_album", linear_state, ALBUM_INSTRUCTION),
        ("B_t2i_album", t2i_state, ALBUM_INSTRUCTION),
        ("C_linear_loop", linear_state, LOOP_INSTRUCTION),
        ("D_t2i_loop", t2i_state, LOOP_INSTRUCTION),
    ]
    rows = []
    for name, state, instruction in cells:
        prefixes = editor.prefixes(instruction)
        base = editor.image_clean(state, read, prefixes["image"])
        edited, _ = edit_state(editor, state, read, prefixes, steps, base_prediction=base)
        displacement = float((edited - state).abs().mean())
        lookahead = preview_image(edited, read, grid, NativeT2I(rt, PARENT, cfg))
        result = euler(edited, tail_times, NativeT2I(rt, PARENT, cfg))
        save_image(args.out / f"{name}_edited_preview.png", lookahead)
        save_image(args.out / f"{name}_result.png", result)
        rows.append(dict(cell=name, instruction=instruction, displacement=displacement))
        print(json.dumps(dict(cell=name, displacement=round(displacement, 5)), ensure_ascii=False),
              flush=True)
    (args.out / "cells.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
