"""Does the first Euler step point at the target? s=0 is the only honest probe.

Training draws ``s ~ U(0,1)`` and builds ``q_s = (1-s) a + s b``, so for any
s>0 the input already contains part of the target and the loss can be lowered by
extrapolating that leak. At ``s = 0`` the input is exactly ``a_t``: none of the
target is present, and the model has to follow the instruction to move at all.
That is also the quantity inference uses for its first step.

For each held-out record this compares the predicted displacement against the
true one, in three ways:

``cosine``   direction agreement; near 1 means it moved the right way;
``ratio``    predicted magnitude over true magnitude; near 1 means the right size;
``relative`` relative error of the vector, so direction and size together.

A model that only extrapolates leaked target shows high cosine but wrong ratio:
it moves along the picture's own structure, not toward the edit.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.images import load_canvas
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor


def _flat(tensor):
    return tensor.detach().float().reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=64)
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
            # s = 0: the state is a_t itself, nothing of the target has entered.
            predicted = editor.state_velocity(anchor, read, 0.0, prefixes["joint"], base)

            a, b = _flat(predicted), _flat(true_speed)
            cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
            true_norm = float(b.norm()) + 1e-8
            ratio = float(a.norm()) / true_norm
            relative = float((a - b).norm()) / true_norm
            # A zero output scores cosine 0 and ratio 0, which is the honest
            # reading: it did not move toward the target.
            row = dict(split=split, uid=record["uid"],
                       instruction=record["edit_instruction"],
                       cosine=cosine, ratio=ratio, relative=relative,
                       seconds=time.perf_counter() - began)
            rows.append(row)
            # Each rank keeps its own file so 64 writers never contend.
            with shard.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(dict(split=split, uid=record["uid"],
                                  cosine=round(cosine, 3), ratio=round(ratio, 3),
                                  relative=round(relative, 3)), ensure_ascii=False), flush=True)

    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    all_rows = []
    for path in sorted(args.out.glob("shard_*.jsonl")):
        all_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    (args.out / "s0_steps.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1))

    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in all_rows if r["split"] == split]
        if not subset:
            continue
        n = len(subset)
        summary[split] = dict(
            cases=n,
            mean_cosine=sum(r["cosine"] for r in subset) / n,
            mean_ratio=sum(r["ratio"] for r in subset) / n,
            mean_relative=sum(r["relative"] for r in subset) / n,
            frac_cosine_above_half=sum(1 for r in subset if r["cosine"] > 0.5) / n,
        )
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
