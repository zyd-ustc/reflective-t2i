"""Split a JSONL manifest into per-rank shards so each worker reads only its own.

Round-robin by line keeps the loader's shard semantics identical while cutting
each process's read volume by ``--world``.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = [(args.out_dir / f"shard_{i:05d}.jsonl").open("w") for i in range(args.world)]
    counts = [0] * args.world
    splits = {i: {} for i in range(args.world)}
    position = 0
    try:
        with args.manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                rank = position % args.world
                position += 1
                handles[rank].write(line)
                counts[rank] += 1
                split = json.loads(line)["split"]
                splits[rank][split] = splits[rank].get(split, 0) + 1
    finally:
        for handle in handles:
            handle.close()

    summary = {i: dict(rows=counts[i], splits=splits[i]) for i in range(args.world)}
    (args.out_dir / "shards.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("total rows:", sum(counts), "across", args.world, "shards")


if __name__ == "__main__":
    main()
