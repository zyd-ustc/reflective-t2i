"""Step 7: training manifest validated directly through LinearManifest."""

import hashlib
import json
from pathlib import Path

from qwen_latent_cot.count_spatial.verify import coarse_edit_region, edit_semantics

PROTOCOL = "semantic_feedback_only_v1"
ROLE = "L_edit"
TASK_CODE = {"count_reduce": "T8", "count_increase": "T8", "spatial": "T9"}


def split_for(family):
    """96/2/2 by family, stable across re-runs of the same uid."""
    value = int(hashlib.sha1(str(family).encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 2 else ("val" if value < 4 else "train")


def record(candidate, plan, verification, instruction, wrong_image, target_image, condition, *, region_min_confidence=0.5):
    family = plan["edit_family"]
    sem = edit_semantics(verification)
    region = coarse_edit_region(verification, min_confidence=region_min_confidence)
    item = dict(
        uid=candidate["uid"],
        split=split_for(candidate["source_family"]),
        source_image=str(wrong_image),
        target_image=str(target_image),
        edit_instruction=instruction,
        tail_condition=condition,
        task=TASK_CODE[family],
        role=ROLE,
        semantic_task=family,
        count_direction=plan["count_direction"],
        source_family=candidate["source_family"],
        prompt_family=candidate["prompt_family"],
        semantic_approved=True,
        condition_protocol=PROTOCOL,
        tail_source="declared",
        benchmark=candidate.get("benchmark"),
        edit_semantics=sem,
        edit_region=region,
        verified=dict(
            passed=verification["passed"],
            target_count=verification.get("target_count"),
            wrong_count=verification.get("wrong_count"),
            checks=verification["checks"],
        ),
    )
    if region["verified"]:
        item["edit_boxes"] = region["boxes"]
    return item


def write_shards(records, out_dir, prefix="manifest", shard_size=500):
    """Fixed-size shards of a JSONL manifest; returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards, buffer = [], []

    def flush(number):
        path = out_dir / ("%s_%03d.jsonl" % (prefix, number))
        with path.open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(item) + "\n" for item in buffer)
        shards.append(path)
        buffer.clear()

    for item in records:
        buffer.append(item)
        if len(buffer) == shard_size:
            flush(len(shards))
    if buffer:
        flush(len(shards))
    return shards


class _ManifestOnlyRuntime:
    identity = None
    width = 512
    height = 512

    def noise_scale(self):
        return 1.0

    def times(self):
        return []


def validate(paths, config_path):
    """Load the shards through LinearManifest so malformed records fail immediately."""
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.data import LinearManifest

    cfg = load_config(str(config_path))
    rt = _ManifestOnlyRuntime()
    linear = LinearManifest([str(p) for p in paths], rt=rt, cfg=cfg)
    splits = {}
    for row in linear.rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    return dict(pairs=len(linear.rows), train=len(linear.train_indices), splits=splits)
