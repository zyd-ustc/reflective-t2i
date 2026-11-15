"""Three zero-training probes, after the confounds in the first eval were found.

The 256-case test had three problems: it kept only counting edits (the hardest
family at t=0.36), it conditioned preview and tail on the instruction text (so
the tail painted "so that it contains exactly 3" as picture content), and it
built the T2I start from that same instruction. None of those say anything
about the editor.

What can say something, cheaply:

``baseline``   the corrected held-out test on train and on val with every edit
               family represented, so the two are comparable.
``lowfreq``    the cosine between the predicted and true displacement after a
               16x blur, where object add/remove/move actually lives.
``oracle``     the native denoiser asked at b_t, which upper-bounds what any
               displacement predictor can recover at this timestep.

The last two also run on the *native IT2I prefix* rather than the shared text
prefix, because that is the condition deployment really has.
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
    if float(a.norm()) < 1e-8 or float(b.norm()) < 1e-8:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=1))


def _low(tensor, factor):
    return F.avg_pool2d(tensor.float(), factor, factor) if factor > 1 else tensor.float()


def _family(text):
    text = (text or '').lower()
    if text.startswith(('add ', 'remove ')):
        return 'count'
    if text.startswith(('rearrange', 'swap', 'move')):
        return 'spatial'
    if text.startswith('replace'):
        return 'replace'
    if text.startswith('change'):
        return 'attribute'
    return 'other'


def _balanced(records, limit, seed):
    """Round-robin over edit families so no single type dominates."""
    buckets = {}
    for record in records:
        buckets.setdefault(_family(record.get('edit_instruction')), []).append(record)
    rng = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda r: r['uid'])
        rng.shuffle(values)
    chosen = []
    while len(chosen) < limit and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(chosen) < limit:
                chosen.append(buckets[family].pop())
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=64)
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
    rows = []
    for split in ("train", "val"):
        pool = _balanced([r for r in records if r["split"] == split], args.per_split, args.seed)
        for position, record in enumerate(pool):
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
            oracle = editor.image_clean(destination, read, prefixes["image"]) - base

            row = dict(
                split=split, family=_family(record["edit_instruction"]), uid=record["uid"],
                cos_full=_cos(predicted, true_speed),
                cos_low=_cos(_low(predicted, args.factor), _low(true_speed, args.factor)),
                oracle_cos_low=_cos(_low(oracle, args.factor), _low(true_speed, args.factor)),
                predicted_over_true=float(predicted.abs().mean()) / (float(true_speed.abs().mean()) + 1e-8),
                seconds=time.perf_counter() - began,
            )
            rows.append(row)
            with shard.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(dict(split=split, family=row["family"],
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
    for split in ("train", "val"):
        subset = [r for r in all_rows if r["split"] == split]
        if not subset:
            continue
        n = len(subset)
        entry = {k: sum(r[k] for r in subset) / n for k in
                 ("cos_full", "cos_low", "oracle_cos_low", "predicted_over_true")}
        entry["cases"] = n
        entry["by_family"] = {}
        for family in sorted({r["family"] for r in subset}):
            group = [r for r in subset if r["family"] == family]
            entry["by_family"][family] = dict(
                cases=len(group),
                cos_low=sum(r["cos_low"] for r in group) / len(group),
                oracle_cos_low=sum(r["oracle_cos_low"] for r in group) / len(group))
        summary[split] = entry
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
