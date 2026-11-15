"""Is the s=0 end the weak part, and is the rest of the range leaking the target?

Training builds ``q_s = (1-s) a + s b`` and draws ``s ~ U(0,1)``. At s>0 the
input already carries part of the target, so a strong base model can lower the
loss by extrapolating what leaked instead of reading the instruction. The only
part of the range that tests instruction following is s≈0, where q_s is a_t and
none of b has entered yet.

This measures the loss in four s buckets on the training split and on the
held-out splits, with the same checkpoint and the same construction. If the
train loss is low everywhere but the held-out loss is only high near s=0, the
end that matters is the end that never got learned.
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
from qwen_latent_cot.u15_target_correction.field import phase_grid
from qwen_latent_cot.u15_target_correction.images import load_canvas
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.loss import state_flow_matching
from qwen_latent_cot.u15_target_correction.model import StateEditor

BUCKETS = ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=64,
                        help="How many records to score per split")
    parser.add_argument("--samples", type=int, default=4,
                        help="s draws per record per bucket (bucket bounds only)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, set_device(args.device), False)
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))
    read = float(cfg["read_time"])
    sigma = rt.noise_scale()
    args.out.mkdir(parents=True, exist_ok=True)

    records = json.loads(args.manifest.read_text())["records"]
    wanted = ("add ", "remove ")
    eligible = [r for r in records
                if (r.get("edit_instruction") or "").lower().startswith(wanted)
                and "exactly" in (r.get("edit_instruction") or "").lower()]

    rng = random.Random(args.seed)
    results = {}
    for split in ("train", "val", "test"):
        pool = sorted([r for r in eligible if r["split"] == split], key=lambda r: r["uid"])
        rng.shuffle(pool)
        selected = pool[:args.per_split]
        sums = {b: 0.0 for b in BUCKETS}
        counts = {b: 0 for b in BUCKETS}
        began = time.perf_counter()
        for record in selected:
            source, _, _ = load_canvas(Path(record["source_image"]), rt.width, rt.height, rt.device)
            target, _, _ = load_canvas(Path(record["target_image"]), rt.width, rt.height, rt.device)
            generator = torch.Generator(device=rt.device).manual_seed(abs(hash(record["uid"])) % (2 ** 31))
            noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                                generator=generator) * sigma
            anchor = read * source + (1 - read) * noise
            destination = read * target + (1 - read) * noise
            valid = torch.ones(1, 1, rt.height, rt.width, device=rt.device)
            prefixes = editor.prefixes(record["edit_instruction"])
            base = editor.image_clean(anchor, read, prefixes["image"])
            for low, high in BUCKETS:
                for _ in range(args.samples):
                    progress = rng.uniform(low, high - 1e-6)
                    state = (1 - progress) * anchor + progress * destination
                    predicted = editor.state_velocity(state, read, progress, prefixes["joint"], base)
                    loss = state_flow_matching(predicted, destination - anchor, read, valid)
                    sums[(low, high)] += float(loss)
                    counts[(low, high)] += 1
        results[split] = {f"{low:.1f}-{high:.1f}": sums[(low, high)] / max(counts[(low, high)], 1)
                          for low, high in BUCKETS}
        results[split]["records"] = len(selected)
        results[split]["seconds"] = round(time.perf_counter() - began, 1)
        print(split, json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                 for k, v in results[split].items()}), flush=True)

    (args.out / "loss_by_s.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
