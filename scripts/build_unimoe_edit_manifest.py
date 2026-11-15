"""Convert UniMoE edit datasets into one linear-pair manifest for Stage 1/2.

Splits are assigned by image family, never by record id: one source image can
carry several instructions, and the loader rejects a family that crosses splits.

Usage:
    python scripts/build_unimoe_edit_manifest.py \
        --root /data/zx_wang_shared_work/ModelScope-zx/clean \
        --out /data/zyd_workspace/isochronic_data_20261107/unimoe_edit/L_edit.jsonl \
        --val-fraction 0.002 --seed 2026
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_latent_cot.u15_target_correction.images import pair_geometry_error

# First-round exclusions: geometry-protocol incompatible, not task-type based.
EXCLUDED_DATASETS = {"HQ-Edit": "source/target aspect ratio differs in 91% of rows"}
GENERATIVE_ONLY = {"OpenGPT-4o-Image_gen"}

# Only the wrappers that sit before the real action are stripped.
TEMPLATE_PREFIX = re.compile(
    r"^(?:image\s+(?:modification|edition|edit)|edit(?:ing)?\s+(?:the\s+)?(?:image|picture)|"
    r"modify\s+the\s+image|picture\s+modification|alter\s+(?:the\s+)?image|"
    r"transform\s+this\s+image|editing\s+task)\s*:\s*",
    re.IGNORECASE,
)

# How the file names map to family names used in reports.
DATASET_FILES = {
    "X2Edit-Dataset": "unimoe2.5_X2Edit-Dataset_clean.json",
    "UltraEdit": "unimoe2.5_UltraEdit_clean.json",
    "SEED-Data-Edit": "unimoe2.5_SEED-Data-Edit_clean.json",
    "AnyEdit": "unimoe2.5_AnyEdit_clean.json",
    "UnicEdit-10M": "unimoe2.5_UnicEdit-10M_clean.json",
    "ImgEdit": "unimoe2.5_ImgEdit_clean.json",
    "instructPix2pix": "unimoe2.5_instructPix2pix_clean.json",
    "HQ-Edit": "unimoe2.5_HQ-Edit_clean.json",
    "OpenGPT-4o-Image_edit": "unimoe2.5_OpenGPT-4o-Image_edit_clean.json",
    "MagicBrush": "unimoe2.5_MagicBrush_clean.json",
}


def stream_objects(path, chunk_bytes=8 << 20):
    """Yield JSON objects from a top-level JSON array, with bounded memory.

    The sources are pretty-printed, one top-level object per line range, so the
    file is split on newlines at depth zero and each slice is parsed on its own.
    orjson parses several times faster than the stdlib decoder.
    """
    try:
        import orjson
        loads = orjson.loads
    except ImportError:  # pragma: no cover - minimal environments
        loads = json.loads

    with open(path, 'rb') as handle:
        buf = b''
        started = False
        depth = 0
        in_string = False
        escape = False
        piece = []
        for chunk in iter(lambda: handle.read(chunk_bytes), b''):
            buf += chunk
            if not started:
                probe = buf.lstrip()
                if not probe:
                    buf = b''
                    continue
                found = probe.find(b'[')
                if found < 0:
                    raise ValueError('expected a JSON array in ' + str(path))
                buf = probe[found + 1:]
                started = True
            last_newline = buf.rfind(b'\n')
            if last_newline < 0:
                if len(buf) > (256 << 20):
                    raise ValueError('line too long in ' + str(path))
                continue
            block = buf[:last_newline]
            buf = buf[last_newline + 1:]
            for line in block.split(b'\n'):
                stripped = line.strip()
                if not stripped:
                    continue
                if depth == 0 and not piece and stripped in (b',', b']', b'['):
                    continue
                piece.append(stripped)
                # Track depth over this line only, excluding string contents.
                i, n = 0, len(stripped)
                while i < n:
                    byte = stripped[i]
                    if in_string:
                        if escape:
                            escape = False
                        elif byte == 0x5C:
                            escape = True
                        elif byte == 0x22:
                            in_string = False
                    elif byte == 0x22:
                        in_string = True
                    elif byte == 0x7B:
                        depth += 1
                    elif byte == 0x7D:
                        depth -= 1
                    i += 1
                if depth == 0 and not in_string and piece:
                    blob = b''.join(piece).strip().rstrip(b',').strip()
                    piece = []
                    if blob and blob not in (b']', b'['):
                        yield loads(blob)
        tail = b''.join(piece).strip().rstrip(b',').strip()
        if tail and tail not in (b']', b'['):
            yield loads(tail)


def normalize_instruction(text):
    """Drop the <image> token and a leading wrapper, keep the real action words."""
    text = str(text or "").replace("<image>", "").strip()
    text = TEMPLATE_PREFIX.sub("", text).strip()
    return re.sub(r"\s+", " ", text)


def source_family(dataset, source_path):
    """A family groups every edit that shares one source image.

    The dataset name is part of the key so that two collections shipping the same
    picture keep separate families; the path itself is normalised of its case.
    """
    key = (dataset + "|" + os.path.normpath(source_path)).lower()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def split_for_family(family, val_fraction, test_fraction):
    """Deterministic family-level split so no family crosses a boundary."""
    bucket = int(family[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def convert_one(dataset, path, stats, rows_out, val_fraction, test_fraction, limit=0):
    seen_ids = set()
    families = defaultdict(set)
    for obj in stream_objects(path):
        if limit and stats["kept"] >= limit:
            break
        stats["read"] += 1
        raw_id = str(obj.get("id") or "").strip()
        uid = dataset + ":" + raw_id
        if not raw_id or uid in seen_ids:
            stats["duplicate_id"] += 1
            continue
        seen_ids.add(uid)

        images = obj.get("image") or []
        source = images[0] if images else None
        target = obj.get("image_gen")
        if not source or not target:
            stats["missing_image_field"] += 1
            continue
        # Existence is checked separately: two remote stat calls per record on
        # shared storage cost far more than parsing the row.

        conversation = obj.get("conversations") or []
        human = next((c.get("value") for c in conversation if c.get("from") == "human"), "")
        instruction = normalize_instruction(human)
        if not instruction:
            stats["empty_instruction"] += 1
            continue

        output_caption = str(obj.get("output_caption") or "").strip()
        if not output_caption:
            stats["empty_output_caption"] += 1
            continue

        family = source_family(dataset, source)
        rows_out.append(
            dict(
                uid=uid,
                source_image=source,
                target_image=target,
                edit_instruction=instruction,
                parent_condition=str(obj.get("input_caption") or "").strip(),
                tail_condition=output_caption,
                source_family=family,
                # Same key as the image family: one source picture and its
                # instructions stay together. Keying this on the instruction text
                # alone split identical wording across train and val.
                prompt_family=family,
                dataset=dataset,
                official_id=raw_id,
                semantic_approved=True,
            )
        )
        families[family].add(dataset)
        stats["kept"] += 1

    for row in rows_out:
        row["split"] = split_for_family(row["source_family"], val_fraction, test_fraction)
    return families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/zx_wang_shared_work/ModelScope-zx/clean")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset names; defaults to every edit set except HQ-Edit")
    parser.add_argument("--val-fraction", type=float, default=0.002)
    parser.add_argument("--test-fraction", type=float, default=0.002)
    parser.add_argument("--limit", type=int, default=0, help="Per-dataset cap, for smoke runs")
    parser.add_argument("--verify-samples", type=int, default=200,
                        help="How many rows per dataset to open for an existence check")
    parser.add_argument("--verify-geometry", action="store_true",
                        help="Open every pair to check the aspect ratio agrees (slow)")
    args = parser.parse_args()

    root = Path(args.root)
    names = args.datasets or [n for n in DATASET_FILES if n not in EXCLUDED_DATASETS]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    totals = Counter()
    per_dataset = {}
    all_rows = []
    for name in names:
        if name in EXCLUDED_DATASETS:
            print("skip %-24s %s" % (name, EXCLUDED_DATASETS[name]))
            continue
        path = root / DATASET_FILES[name]
        if not path.exists():
            print("MISSING", path)
            continue
        stats = Counter()
        rows = []
        convert_one(name, path, stats, rows, args.val_fraction, args.test_fraction, args.limit)

        if args.verify_geometry:
            from PIL import Image

            kept = []
            for row in rows:
                with Image.open(row["source_image"]) as a, Image.open(row["target_image"]) as b:
                    reason = pair_geometry_error(row, a.size, b.size)
                if reason:
                    stats["geometry_rejected"] += 1
                    continue
                kept.append(row)
            rows = kept

        missing = sum(
            1 for row in rows[: args.verify_samples]
            if not (os.path.exists(row["source_image"]) and os.path.exists(row["target_image"]))
        )
        if missing:
            stats["missing_image_file_sampled"] = missing
        per_dataset[name] = dict(stats, kept=len(rows))
        totals.update(stats)
        totals["kept"] = totals.get("kept", 0) + len(rows)
        all_rows.extend(rows)
        print("%-24s read=%-9d kept=%-9d families=%d" % (
            name, stats["read"], len(rows), len({r["source_family"] for r in rows})))

    # A family must not cross splits; the loader enforces this too.
    family_splits = defaultdict(set)
    for row in all_rows:
        family_splits[row["source_family"]].add(row["split"])
    crossing = [f for f, s in family_splits.items() if len(s) > 1]
    if crossing:
        raise SystemExit("internal error: %d families cross splits" % len(crossing))

    with args.out.open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_split = Counter(row["split"] for row in all_rows)
    by_dataset = Counter(row["dataset"] for row in all_rows)
    report = dict(
        output=str(args.out),
        rows=len(all_rows),
        families=len(family_splits),
        splits=dict(by_split),
        by_dataset=dict(by_dataset),
        filters=dict(totals),
        excluded=EXCLUDED_DATASETS,
    )
    (args.out.parent / (args.out.stem + ".report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print("\nrows=%d families=%d splits=%s" % (len(all_rows), len(family_splits), dict(by_split)))
    print("filters:", dict(totals))


if __name__ == "__main__":
    main()
