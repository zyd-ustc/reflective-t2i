"""Verify every pair in a manifest shares one scene extent, in parallel.

Writes a JSONL file of rejected rows plus a summary, and exits nonzero when any
pair is geometrically incompatible.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_latent_cot.u15_target_correction.images import pair_geometry_error


def _size(path):
    """Read only the header, never the pixels: the shared storage is the bottleneck."""
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _check(row):
    source, target = row.get("source_image"), row.get("target_image")
    try:
        reason = pair_geometry_error(row, _size(source), _size(target))
    except Exception as error:  # unreadable file
        reason = type(error).__name__ + ": " + str(error)[:120]
    return (row.get("uid"), reason)


def rows_of(path):
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        for row in (data["records"] if isinstance(data, dict) else data):
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = rejected = 0
    with args.out.open("w") as bad, ProcessPoolExecutor(max_workers=args.workers) as pool:
        batch = []
        for row in rows_of(args.manifest):
            batch.append(row)
            if args.limit and total + len(batch) >= args.limit:
                batch = batch[: args.limit - total]
            if len(batch) >= 2000:
                for uid, reason in pool.map(_check, batch, chunksize=16):
                    total += 1
                    if reason:
                        rejected += 1
                        bad.write(json.dumps({"uid": uid, "reason": reason}) + "\n")
                batch = []
                print("checked %d, rejected %d" % (total, rejected), flush=True)
            if args.limit and total >= args.limit:
                break
        if batch:
            for uid, reason in pool.map(_check, batch, chunksize=16):
                total += 1
                if reason:
                    rejected += 1
                    bad.write(json.dumps({"uid": uid, "reason": reason}) + "\n")

    summary = dict(manifest=str(args.manifest), checked=total, rejected=rejected,
                   rate=round(rejected / total, 6) if total else 0.0)
    (args.out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
