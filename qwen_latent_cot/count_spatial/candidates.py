"""Step 1: read the verified T2I library and pick images worth editing.

Suitability is decided by the observation step, not here. This step only
applies the rules that are cheap and certain: the library's benchmark set, an
image that exists, and an image the training loader can pair (a square-ish
canvas; the pair shares one rectangle).
"""

import csv
import hashlib
import json
from pathlib import Path

# Names as the library actually spells them, not the shortened plan spellings.
BENCHMARKS = ("geneval2", "dpg", "t2i_compbench_pp", "spatial_geneval", "oneig_bench", "tiif_bench")
# Wide or tall images are cropped to a shared canvas; beyond this they waste it.
MAX_RATIO = 2.01


def _split(uid):
    value = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 2 else ("val" if value < 4 else "train")


def read_index(index_csv, benchmarks=BENCHMARKS, max_ratio=MAX_RATIO, need_png=True):
    """Yield one candidate dict per usable index row, in file order."""
    with open(index_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["benchmark"] not in benchmarks:
                continue
            width, height = int(row["width"]), int(row["height"])
            ratio = max(width, height) / min(width, height)
            if ratio > max_ratio:
                continue
            image = Path(row["image"])
            if not image.exists():
                continue
            if need_png and image.suffix.lower() != ".png":
                continue
            uid = row["uid"]
            yield dict(
                uid=uid,
                benchmark=row["benchmark"],
                wh_ratio=row["wh_ratio"],
                width=width,
                height=height,
                image=str(image),
                source_family=uid,
                prompt_family=uid,
                split=_split(uid),
            )


def select(index_csv, limit, out_path, benchmarks=BENCHMARKS, max_ratio=MAX_RATIO, seed=2026, skip=0):
    """A seeded shuffle of the usable rows, balanced over edit families later.

    The seed makes trials reproducible, and because the shuffle covers the
    whole library, ``skip`` lets the next trial start past the last one's slice
    instead of re-proposing the same images.
    """
    import random

    rows = list(read_index(index_csv, benchmarks, max_ratio))
    random.Random(seed).shuffle(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, per_benchmark = 0, {}
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows[skip:]:
            if kept >= limit:
                break
            handle.write(json.dumps(row) + "\n")
            kept += 1
            per_benchmark[row["benchmark"]] = per_benchmark.get(row["benchmark"], 0) + 1
    return dict(written=kept, usable=len(rows), per_benchmark=per_benchmark)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
