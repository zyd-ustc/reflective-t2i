"""Two checks before blaming the model for a cosine of 0.57.

1. Low-frequency cosine. Object-level edits (add, remove, move) live in the low
   frequencies; per-pixel agreement is dominated by texture the editor cannot
   be expected to match. Downsampling both directions answers whether the
   semantic direction was learned even when the high-frequency match is poor.

2. The oracle ceiling. Ask the frozen native denoiser what it would predict at
   ``b_t`` against the same source-minus-target difference. That is the best any
   displacement predictor can do on this target at this timestep, because the
   native model has no instruction to interpret and already sits at the target.
   If the ceiling is around 0.65, then 0.575 is close to it and "cos > 0.9" was
   never a reachable goal.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.images import load_canvas
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor


def _cos(a, b):
    a, b = a.detach().float().reshape(1, -1), b.detach().float().reshape(1, -1)
    return float(F.cosine_similarity(a, b, dim=1))


def _lowpass(tensor, factor=16):
    """Average pooling keeps the layout and drops the per-pixel texture."""
    if factor <= 1:
        return tensor.float()
    return F.avg_pool2d(tensor.float(), factor, factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=32)
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    from qwen_latent_cot.u15_target_correction import distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = set_device(args.device) if world < 2 else dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, world > 1)
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))
    read = float(cfg["read_time"])
    sigma = rt.noise_scale()
    args.out.mkdir(parents=True, exist_ok=True)
    shard = args.out / f"shard_{rank:02d}.jsonl"

    records = json.loads(args.manifest.read_text())["records"]
    rng = random.Random(args.seed)
    rows = []
    for split in ("train", "val", "test"):
        pool = sorted([r for r in records if r["split"] == split], key=lambda r: r["uid"])
        rng.shuffle(pool)
        for position, record in enumerate(pool[:args.per_split]):
            if position % world != rank:
                continue
            began = time.perf_counter()
            source, _, _ = load_canvas(Path(record["source_image"]), rt.width, rt.height, rt.device)
            target, _, _ = load_canvas(Path(record["target_image"]), rt.width, rt.height, rt.device)
            generator = torch.Generator(device=rt.device).manual_seed(abs(hash(record["uid"])) % (2 ** 31))
            noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                                generator=generator) * sigma
            anchor = read * source + (1 - read) * noise
            destination = read * target + (1 - read) * noise
            true_speed = destination - anchor

            prefixes = editor.prefixes(record["edit_instruction"])
            base = editor.image_clean(anchor, read, prefixes["image"])
            predicted = editor.state_velocity(anchor, read, 0.0, prefixes["joint"], base)

            # The oracle: the frozen native denoiser asked at b_t. It has no
            # instruction, so whatever it recovers is the structural ceiling.
            native_at_b = editor.image_clean(destination, read, prefixes["image"])
            oracle_speed = native_at_b - base

            low_pred, low_true = _lowpass(predicted, args.factor), _lowpass(true_speed, args.factor)
            low_oracle = _lowpass(oracle_speed, args.factor)

            row = dict(
                split=split, uid=record["uid"],
                cos_full=_cos(predicted, true_speed),
                cos_low=_cos(low_pred, low_true),
                oracle_cos_full=_cos(oracle_speed, true_speed),
                oracle_cos_low=_cos(low_oracle, low_true),
                # How much of the true low-frequency direction the editor got,
                # normalised by what the oracle got on the same record.
                oracle_headroom=float(low_oracle.norm()) / (float(low_true.norm()) + 1e-8),
                seconds=time.perf_counter() - began,
            )
            rows.append(row)
            with shard.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(dict(split=split, cos_full=round(row["cos_full"], 3),
                                  cos_low=round(row["cos_low"], 3),
                                  oracle_low=round(row["oracle_cos_low"], 3)),
                             ensure_ascii=False), flush=True)

    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    all_rows = []
    for path in sorted(args.out.glob("shard_*.jsonl")):
        all_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    (args.out / "rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1))

    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in all_rows if r["split"] == split]
        if not subset:
            continue
        n = len(subset)
        summary[split] = {k: sum(r[k] for r in subset) / n for k in
                          ("cos_full", "cos_low", "oracle_cos_full", "oracle_cos_low",
                           "oracle_headroom")}
        summary[split]["cases"] = n
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
