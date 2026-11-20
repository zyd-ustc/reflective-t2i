"""Linear edit-pair dataset and shuffled epoch sampler for isochronic manifold training."""

import hashlib
import json
import random
from pathlib import Path

from . import CONDITION_PROTOCOL, LINEAR_PAIR, ROLE_L_EDIT
from .conditions import compact
from .images import load_pair


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read(path):
    return json.loads(Path(path).read_text())


def _stream_rows(path):
    """Yield rows from a JSONL file without materialising the whole file."""
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_rows(path):
    """Prompt rows from JSONL, a JSON list, or a ``records`` object."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return list(_stream_rows(path))
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    raise ValueError("Prompt file must be a list, records object, or JSONL")


def count_index_shards(path):
    """How many per-rank index shards a manifest declares, or 0 when it is not sharded."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return 0
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get("shards"):
        return len(data["shards"])
    return 0


def _approved(row):
    value = row.get("semantic_approved", row.get("approved", True))
    return bool(value)


def _remember(groups, key, split):
    if key is None:
        return
    previous = groups.setdefault(str(key), split)
    if previous != split:
        raise ValueError("A prompt or source family crosses splits")


def _family(row, key, fallback):
    return row.get(key) or fallback


def classify_semantic_task(instruction, task=None):
    """Derive semantic task family and count direction for logging and evaluation."""
    text = compact(instruction).lower()
    direction = None
    if text.startswith("add "):
        return "count", "add"
    if text.startswith("remove "):
        return "count", "remove"
    if task == "T8":
        if "add" in text and "remove" not in text:
            direction = "add"
        elif "remove" in text and "add" not in text:
            direction = "remove"
        return "count", direction
    if task == "T9" or text.startswith(("rearrange", "swap", "move")):
        return "spatial", None
    if task in ("T3", "T4") or text.startswith("replace"):
        return "replace", None
    if task in ("T2", "T5") or text.startswith("change"):
        return "attribute", None
    return "other", None


def _coerce_paths(paths):
    if isinstance(paths, (str, Path)):
        items = [p.strip() for p in str(paths).split(",") if p.strip()]
    else:
        items = [str(p).strip() for p in paths if str(p).strip()]
    if not items:
        raise ValueError("At least one manifest path is required")
    return [Path(p).resolve() for p in items]


class LinearManifest:
    """One or more linear edit-pair manifests consumed by isochronic manifold training."""

    def __init__(self, path, rt, cfg, rank=None, world=1):
        self.paths = _coerce_paths(path)
        self.path = self.paths[0]
        self.root = self.path.parent
        self.rt, self.cfg = rt, cfg
        self.rows = []
        self.rank = None if rank is None else int(rank)
        self.world = max(1, int(world))
        prompt_splits, source_splits = {}, {}
        for manifest_path in self.paths:
            root = manifest_path.parent
            if manifest_path.suffix == ".jsonl":
                records = _stream_rows(manifest_path)
            else:
                manifest = _read(manifest_path)
                if "records" not in manifest:
                    raise ValueError("A linear manifest needs a records list: " + str(manifest_path))
                records = manifest["records"]
            for global_index, raw in enumerate(records):
                if self.rank is not None and global_index % self.world != self.rank:
                    continue
                row = self._normalize(raw, root)
                row["global_index"] = global_index
                _remember(prompt_splits, row["prompt_family"], row["split"])
                _remember(source_splits, row["source_family"], row["split"])
                self.rows.append(row)
        self.train_indices = [i for i, row in enumerate(self.rows) if row["split"] == "train"]
        if not self.train_indices:
            raise ValueError("The linear manifest has no training rows")

    @property
    def count(self):
        return len(self.train_indices)

    def _normalize(self, row, root):
        role = row.get("role")
        if role not in (None, "native_edit", LINEAR_PAIR, ROLE_L_EDIT):
            raise ValueError("A linear manifest only carries forward edit pairs")
        split = row.get("split")
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val, or test")
        if not _approved(row):
            raise ValueError("Unapproved edit pairs cannot enter training")
        uid = str(row.get("uid") or "").strip()
        if not uid:
            raise ValueError("Every linear pair needs a uid")
        source_image = row.get("source_image")
        target_image = row.get("target_image")
        if not source_image or not target_image:
            raise ValueError("Every linear pair needs source and target images")
        if not _resolve(root, source_image).exists() or not _resolve(root, target_image).exists():
            raise ValueError("A linear pair image is missing: " + uid)
        edit = row.get("edit_instruction") or row.get("edit_condition") or row.get("instruction")
        if not str(edit or "").strip():
            raise ValueError("A linear pair needs a short edit instruction: " + uid)
        edit = compact(edit)
        raw_proto = row.get("condition_protocol")
        if raw_proto not in (None, CONDITION_PROTOCOL, "isochronic_condition_v2", "derived_parent_plus_edit"):
            raise ValueError("Linear condition protocol differs from this run: " + uid)
        semantic_task, count_direction = classify_semantic_task(edit, row.get("task"))
        return dict(
            uid=uid,
            split=split,
            source_image=str(_resolve(root, source_image)),
            target_image=str(_resolve(root, target_image)),
            source_family=_family(
                row, "source_family", hashlib.sha1(str(source_image).encode()).hexdigest()[:16]
            ),
            prompt_family=_family(row, "prompt_family", uid),
            edit_instruction=edit,
            condition_protocol=CONDITION_PROTOCOL,
            task=row.get("task"),
            semantic_task=row.get("semantic_task") or semantic_task,
            count_direction=row.get("count_direction") or count_direction,
            geometry=row.get("geometry"),
            role=ROLE_L_EDIT,
        )

    def choose_time(self, rng, time=None):
        if time is not None:
            return float(time)
        return float(self.cfg["read_time"])

    def sample(self, index, time, keep=False):
        row = self.rows[index]
        source, target, valid, _ = load_pair(
            row, self.rt.width, self.rt.height, self.rt.device
        )
        return dict(
            uid=row["uid"],
            source_kind=LINEAR_PAIR,
            keep=False,
            source=source,
            target=target,
            valid=valid,
            time=float(time),
            edit_request=row["edit_instruction"],
            condition_protocol=row["condition_protocol"],
            source_family=row["source_family"],
            prompt_family=row["prompt_family"],
            task=row["task"],
            semantic_task=row["semantic_task"],
            count_direction=row["count_direction"],
            role=ROLE_L_EDIT,
        )


class TrainingSampler:
    """Traverse every training record once per epoch in a shuffled, non-overlapping order."""

    def __init__(self, linear, cfg, rank=0, world=1):
        self.cfg = cfg
        self.rank = int(rank)
        self.world = max(1, int(world))
        self.rng = random.Random(int(cfg["seed"]) + int(rank))
        self.epoch_rng = random.Random(int(cfg["seed"]) * 7 + 1)
        self.draws = 0
        self.pools = {ROLE_L_EDIT: linear}
        self.weights = {ROLE_L_EDIT: 1.0}
        self.configured = {ROLE_L_EDIT: 1.0}
        if cfg.get("task_weights"):
            raise ValueError(
                "task_weights is retired: traversal is record-equal and shuffled. "
                "Drop task_weights from the configuration."
            )
        self.indices = list(linear.train_indices)
        if not self.indices:
            raise ValueError("The linear manifest has no training rows")
        self.epoch = 0
        self.cursor = 0
        self.order = []
        self._shuffle()
        self.seen = set()

    def _shuffle(self):
        order = list(self.indices)
        self.epoch_rng.shuffle(order)
        self.order = order
        self.cursor = 0

    @property
    def configured_weights(self):
        return dict(self.configured)

    @property
    def actual_weights(self):
        return dict(self.weights)

    @property
    def size(self):
        return len(self.indices)

    def input_domains(self):
        return ["linear"] if self.seen else []

    def draw(self, time=None):
        if self.cursor >= len(self.order):
            self.epoch += 1
            self._shuffle()
        index = self.order[self.cursor]
        self.cursor += 1
        pool = self.pools[ROLE_L_EDIT]
        chosen_time = pool.choose_time(self.rng, time)
        sample = pool.sample(index, chosen_time, keep=False)
        sample["role"] = ROLE_L_EDIT
        sample["epoch"] = self.epoch
        self.seen.add(sample["role"])
        self.draws += 1
        return sample

    def state_dict(self):
        return dict(
            rng=self.rng.getstate(),
            epoch_rng=self.epoch_rng.getstate(),
            epoch=self.epoch,
            cursor=self.cursor,
            order=list(self.order),
            seen=sorted(self.seen),
            draws=self.draws,
            keep=0,
        )

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        if state.get("epoch_rng") is not None:
            self.epoch_rng.setstate(state["epoch_rng"])
        order = state.get("order")
        if order is not None:
            self.order = list(order)
        else:
            self._shuffle()
        self.epoch = int(state.get("epoch", 0))
        self.cursor = int(state.get("cursor", 0))
        self.seen = set(state.get("seen", []))
        self.draws = int(state.get("draws", 0))


def data_version(*paths):
    digest = hashlib.sha256()
    resolved = []
    for item in paths:
        if item is None:
            continue
        if isinstance(item, (list, tuple)):
            resolved.extend(Path(sub).resolve() for sub in item)
        else:
            resolved.append(Path(item).resolve())
    for path in sorted(resolved):
        digest.update(str(path).encode())
        files = sorted(path.glob("manifest*.json")) if path.is_dir() else [path]
        for file in files:
            digest.update(str(file).encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()
