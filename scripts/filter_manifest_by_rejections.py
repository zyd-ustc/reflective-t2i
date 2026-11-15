"""Drop the rows a verification pass rejected, keeping families on one side."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rejected = set()
    with args.rejected.open() as handle:
        for line in handle:
            if line.strip():
                rejected.add(json.loads(line)["uid"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    family_splits = defaultdict(set)
    splits = Counter()
    with args.out.open("w") as out:
        with args.manifest.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["uid"] in rejected:
                    dropped += 1
                    continue
                family_splits[row["source_family"]].add(row["split"])
                splits[row["split"]] += 1
                kept += 1
                out.write(line)

    crossing = sum(1 for s in family_splits.values() if len(s) > 1)
    summary = dict(kept=kept, dropped=dropped, families_crossing_splits=crossing,
                   splits=dict(splits))
    (args.out.parent / (args.out.stem + ".filter.json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 1 if crossing else 0


if __name__ == "__main__":
    raise SystemExit(main())
