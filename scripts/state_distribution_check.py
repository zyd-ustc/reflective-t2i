"""Is the editor blind to real trajectory states, or only to its training states?

Training built every start as ``t * x_src + (1 - t) * sigma * eps`` on a clean
source image. Deployment hands the editor the state a T2I trajectory actually
holds at the read time. Those are two different distributions, and the closed
loop has been failing while the linear route was never the one being tested.

For each prompt this measures both, with the same instruction and the same
checkpoint:

``t2i``     the state a T2I walk holds at the read time (what deployment has);
``linear``  the state built the way training built it, from the same completion.

A displacement norm and the finished picture are saved for each, so the answer
is not only a number.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, edit_state, euler, phase_grid, preview_image, segment_times,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor

# Instructions taken from the reflection run on these prompts, so the wording is
# the one the loop actually produced rather than a hand-written ideal.
DEFAULT_CASES = [
    ("seven green croissants",
     "remove two green croissants so that nine becomes seven, keeping the remaining seven unchanged"),
    ("four rabbits and a sheep",
     "add two rabbits so that 2 becomes 4, keeping the existing two rabbits and two sheep unchanged"),
    ("five bears and a donut",
     "add one brown bear so that 4 bears become 5, placing the new bear in the empty space among the existing bears"),
    ("a car and six flowers",
     "remove one flower so that 7 becomes 6, keeping the car and the remaining six flowers unchanged"),
    ("seven black cows",
     "add one black cow so that 6 becomes 7, in the empty space between Cow 3 and Cow 4 in the middle of the grassy area"),
    ("a plastic motorcycle and a clock",
     "add a plastic motorcycle so that 0 becomes 1, placing it in the empty space to the right of the clock"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", type=Path, help="JSON list of [prompt, instruction]")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=2026)
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

    cases = json.loads(args.cases.read_text()) if args.cases else DEFAULT_CASES

    rows = []
    for index, (prompt, instruction) in enumerate(cases):
        began = time.perf_counter()
        generator = torch.Generator(device=rt.device).manual_seed(args.seed + index)
        noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                            generator=generator) * sigma

        # The deployment route: walk T2I to the read time and remember the state.
        native = NativeT2I(rt, prompt, cfg)
        t2i_state = euler(noise.float(), segment_times(grid, 0.0, read), native)
        completion = euler(t2i_state.clone(), tail_times, NativeT2I(rt, prompt, cfg))
        # The training route: t * x_src + (1 - t) * sigma * eps, exactly as
        # train._linear_fields builds it, with the same noise draw so the only
        # difference from the T2I state is how the state was made.
        linear_state = read * completion + (1 - read) * noise

        saved = {}
        for tag, state in (("t2i", t2i_state), ("linear", linear_state)):
            prefixes = editor.prefixes(instruction)
            base = editor.image_clean(state, read, prefixes["image"])
            edited, _ = edit_state(editor, state, read, prefixes, steps, base_prediction=base)
            displacement = float((edited - state).abs().mean())
            # The state shift in the units the training target used: the whole
            # edit should be about one source->target step, i.e. (b-a)/t.
            equivalent = float(((edited - state) / read).abs().mean())
            result = euler(edited, tail_times, NativeT2I(rt, prompt, cfg))
            save_image(args.out / f"{index:02d}_{tag}_result.png", result)
            save_image(args.out / f"{index:02d}_{tag}_preview.png",
                       preview_image(edited, read, grid, NativeT2I(rt, prompt, cfg)))
            saved[tag] = dict(displacement=displacement, equivalent_source_shift=equivalent)
        save_image(args.out / f"{index:02d}_completion.png", completion)
        save_image(args.out / f"{index:02d}_t2i_state.png", t2i_state)
        save_image(args.out / f"{index:02d}_linear_state.png", linear_state)

        # How far apart the two starts are, in the same units.
        separation = float((t2i_state - linear_state).abs().mean())
        row = dict(index=index, prompt=prompt, instruction=instruction,
                   state_separation=separation, **{f"{k}_{t}": v for t, d in saved.items()
                                                   for k, v in d.items()},
                   seconds=time.perf_counter() - began)
        rows.append(row)
        (args.out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
        print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                          for k, v in row.items() if k != "instruction"}, ensure_ascii=False),
              flush=True)

    t2i = [r["displacement_t2i"] for r in rows]
    lin = [r["displacement_linear"] for r in rows]
    summary = dict(cases=len(rows),
                   mean_displacement_t2i=sum(t2i) / len(t2i),
                   mean_displacement_linear=sum(lin) / len(lin),
                   mean_state_separation=sum(r["state_separation"] for r in rows) / len(rows))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
