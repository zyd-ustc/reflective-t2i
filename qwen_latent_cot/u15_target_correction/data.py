"""Linear image pairs, task-weighted sampling, and tail-generator datasets."""

import hashlib
import json
import random
from pathlib import Path

import torch

from . import LINEAR_PAIR, ROLE_L_EDIT, ROLE_L_REVERSE
from .conditions import compact
from .field import protocol, sample_state_time, weighted_choice
from .images import load_canvas, load_pair


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
    """Derive semantic task family and count direction for logging and task weighting."""
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


def _keep_or_edit(row, cfg, keep):
    """The instruction triple for a keep draw or for the record's own edit."""
    if keep:
        return cfg["keep_instruction"], row["parent_condition"], ""
    return row["edit_instruction"], row["tail_condition"], row["preserve"]


# Keys early caches carried that describe the draw, not the content.
RETIRED_PROTOCOL_KEYS = frozenset({"state_times", "state_time_probs"})


def protocol_matches(cached, bound):
    """Whether a cached manifest was produced by this run's discretisation."""
    if not isinstance(cached, dict):
        return False
    compare = {k: v for k, v in cached.items() if k not in RETIRED_PROTOCOL_KEYS}
    if "phase_grid_times" not in compare:
        compare["phase_grid_times"] = [float(t) for t in cached.get("state_times", [])]
    return compare == bound


def _coerce_paths(paths):
    if isinstance(paths, (str, Path)):
        items = [p.strip() for p in str(paths).split(",") if p.strip()]
    else:
        items = [str(p).strip() for p in paths if str(p).strip()]
    if not items:
        raise ValueError("At least one manifest path is required")
    return [Path(p).resolve() for p in items]


def read_captions(paths):
    """Merge uid -> caption JSONL files into one lookup."""
    captions = {}
    for path in _coerce_paths(paths):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                uid = str(row.get("uid") or "").strip()
                text = compact(
                    row.get("tail_condition")
                    or row.get("target_caption")
                    or row.get("caption")
                    or row.get("parent_condition")
                )
                if uid and text:
                    captions[uid] = text
    if not captions:
        raise ValueError("The caption files carry no uid -> caption rows")
    return captions


class LinearManifest:
    """One or more linear edit-pair manifests consumed directly as L data."""

    def __init__(self, path, rt, cfg, captions=None, rank=None, world=1):
        self.paths = _coerce_paths(path)
        self.path = self.paths[0]
        self.root = self.path.parent
        self.rt, self.cfg = rt, cfg
        self.captions_paths = _coerce_paths(captions) if captions else []
        self.captions = read_captions(self.captions_paths) if self.captions_paths else {}
        self.bound = protocol(rt, cfg)
        self.rows = []
        # Each rank keeps only its stride of the global order. The sampler walks
        # every record once per epoch, so a shard never loses coverage; it keeps
        # 64 ranks from each slurping a multi-gigabyte manifest.
        self.rank = None if rank is None else int(rank)
        self.world = max(1, int(world))
        prompt_splits, source_splits = {}, {}
        for manifest_path in self.paths:
            root = manifest_path.parent
            if manifest_path.suffix == ".jsonl":
                # A converted full-pool manifest streams; it carries no protocol header.
                records = _stream_rows(manifest_path)
            else:
                manifest = _read(manifest_path)
                if "records" not in manifest:
                    raise ValueError("A linear manifest needs a records list: " + str(manifest_path))
                if manifest.get("protocol") is not None and not protocol_matches(manifest["protocol"], self.bound):
                    raise ValueError("Linear manifest protocol differs from this run: " + str(manifest_path))
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
        """Records in this shard (already reported per rank in training logs)."""
        return len(self.train_indices)

    def _normalize(self, row, root):
        role = row.get("role")
        allowed_linear = (None, "native_edit", LINEAR_PAIR, ROLE_L_EDIT, ROLE_L_REVERSE)
        if role not in allowed_linear:
            raise ValueError("A linear manifest only carries edit pairs")
        if role != ROLE_L_REVERSE:
            role = ROLE_L_EDIT
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
        parent = compact(row.get("parent_condition") or row.get("raw_prompt") or row.get("source_prompt"))
        tail = compact(
            row.get("tail_condition")
            or row.get("target_caption")
            or row.get("prompt")
            or self.captions.get(uid)
        )
        if not tail:
            raise ValueError(
                "A linear pair needs a target caption for the tail generator: " + uid
            )
        tail_source = str(row.get("tail_source") or "declared")
        frozen = row.get("condition_protocol") or self.cfg["condition_protocol"]
        if frozen == "derived_parent_plus_edit":
            if not self.cfg.get("allow_derived_tail"):
                raise ValueError(
                    "Derived parent+edit tails are disabled for this stage: " + uid
                )
            tail_source = "derived_parent_plus_edit"
            frozen = self.cfg["condition_protocol"]
        elif frozen != self.cfg["condition_protocol"]:
            raise ValueError("Linear condition protocol differs from this run: " + uid)
        semantic_task, count_direction = classify_semantic_task(edit, row.get("task"))
        return dict(
            uid=uid,
            split=split,
            source_image=str(_resolve(root, source_image)),
            target_image=str(_resolve(root, target_image)),
            source_family=_family(row, "source_family", hashlib.sha1(str(source_image).encode()).hexdigest()[:16]),
            prompt_family=_family(row, "prompt_family", uid),
            edit_instruction=edit,
            parent_condition=parent,
            tail_condition=tail,
            preserve=compact(row.get("preserve")),
            condition_protocol=frozen,
            tail_source=tail_source,
            semantic_quality_source=row.get("semantic_quality_source", "legacy_semantic_approved"),
            target_quality_source=row.get("target_quality_source", "legacy_semantic_approved"),
            task=row.get("task"),
            semantic_task=row.get("semantic_task") or semantic_task,
            count_direction=row.get("count_direction") or count_direction,
            geometry=row.get("geometry"),
            role=role,
        )

    @property
    def count(self):
        return len(self.train_indices)

    def role_pools(self):
        """Forward and reversed pairs are sampled apart even though both are linear."""
        grouped = {}
        for index in self.train_indices:
            grouped.setdefault(self.rows[index]["role"], []).append(index)
        return {
            "linear": self,
            ROLE_L_EDIT: LinearPool(self, grouped.get(ROLE_L_EDIT, [])),
            ROLE_L_REVERSE: LinearPool(self, grouped.get(ROLE_L_REVERSE, [])),
        }

    def available_times(self):
        return [float(t) for t in self.cfg["state_times"]]

    def choose_time(self, rng, time=None):
        if time is not None:
            if float(time) not in self.available_times():
                raise ValueError("Unsupported linear state time")
            return float(time)
        return sample_state_time(self.cfg, rng)

    def choose_index(self, time, rng):
        return rng.choice(self.train_indices)

    def sample(self, index, time, keep=False):
        row = self.rows[index]
        if keep:
            declared = row.get("geometry") or {}
            crop = declared.get("source") if isinstance(declared, dict) else None
            source, valid, _ = load_canvas(
                row["source_image"], self.rt.width, self.rt.height, self.rt.device,
                crop=None if crop in (None, "full") else crop,
            )
            target = source.clone()
            frozen = self.cfg["condition_protocol"]
        else:
            source, target, valid, _ = load_pair(
                row, self.rt.width, self.rt.height, self.rt.device
            )
            frozen = row["condition_protocol"]
        edit, tail, preserve = _keep_or_edit(row, self.cfg, keep)
        return dict(
            uid=row["uid"],
            source_kind=LINEAR_PAIR,
            keep=bool(keep),
            source=source,
            target=target,
            state_a=None,
            state_b=None,
            preview=None,
            base_prediction=None,
            valid=valid,
            time=float(time),
            edit_request=edit,
            parent_condition=row["parent_condition"],
            tail_condition=tail,
            preserve=preserve,
            condition_protocol=frozen,
            tail_source=row["tail_source"],
            source_family=row["source_family"],
            prompt_family=row["prompt_family"],
            semantic_quality_source=row["semantic_quality_source"],
            target_quality_source=row["target_quality_source"],
            task=row["task"],
            semantic_task=row["semantic_task"],
            count_direction=row["count_direction"],
            role=row["role"],
        )


class LinearPool:
    """One sub-role of the linear manifest, with the same interface as a state pool."""

    def __init__(self, manifest, indices):
        self.manifest = manifest
        self.indices = list(indices)
        self.rows = manifest.rows
        self.cfg = manifest.cfg

    @property
    def count(self):
        return len(self.indices)

    def available_times(self):
        return self.manifest.available_times()

    def choose_time(self, rng, time=None):
        return self.manifest.choose_time(rng, time)

    def choose_index(self, time, rng):
        if not self.indices:
            raise ValueError("An empty linear sub-role pool was sampled")
        return rng.choice(self.indices)

    def sample(self, index, time, keep=False):
        return self.manifest.sample(index, time, keep=keep)


class TrainingSampler:
    """Traverse every training record once per epoch in a shuffled, non-overlapping order.

    Records are equally weighted: no dataset or task is up-sampled. Each rank
    owns a disjoint stride of the shuffled order, so ``world_size`` steps of the
    sampler consume ``world_size`` distinct records. The cursor is part of the
    state dict, so a resume continues mid-epoch instead of restarting it.
    """

    def __init__(self, linear, cfg, rank=0, world=1):
        self.cfg = cfg
        self.rank = int(rank)
        self.world = max(1, int(world))
        self.rng = random.Random(int(cfg["seed"]) + int(rank))
        self.epoch_rng = random.Random(int(cfg["seed"]) * 7 + 1)
        self.keep_draws = 0
        self.draws = 0
        self.pools = {ROLE_L_EDIT: linear}
        self.weights = {ROLE_L_EDIT: 1.0}
        self.configured = {ROLE_L_EDIT: 1.0}
        self.keep_fraction = float(cfg.get("keep_fraction", 0.0))
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
        keep = self.keep_fraction > 0 and self.rng.random() < self.keep_fraction
        chosen_time = pool.choose_time(self.rng, time)
        sample = pool.sample(index, chosen_time, keep=keep)
        sample["role"] = ROLE_L_EDIT
        sample["epoch"] = self.epoch
        self.seen.add(sample["role"])
        self.draws += 1
        self.keep_draws += int(bool(sample.get("keep")))
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
            keep=self.keep_draws,
        )

    def load_state_dict(self, state):
        self.rng.setstate(state["rng"])
        if state.get("epoch_rng") is not None:
            self.epoch_rng.setstate(state["epoch_rng"])
        # Restore the very order that was being walked, not a fresh shuffle.
        order = state.get("order")
        if order is not None:
            self.order = list(order)
        else:
            self._shuffle()
        self.epoch = int(state.get("epoch", 0))
        self.cursor = int(state.get("cursor", 0))
        self.seen = set(state.get("seen", []))
        self.draws = int(state.get("draws", 0))
        self.keep_draws = int(state.get("keep", 0))


class TailManifest:
    """Manifest of target images, optional cached q_K states, and T2I captions."""

    def __init__(self, path, rt, cfg, rank=None, world=1):
        self.paths = _coerce_paths(path)
        self.path = self.paths[0]
        self.rt, self.cfg = rt, cfg
        self.captions_paths = []
        self.rank = None if rank is None else int(rank)
        self.world = max(1, int(world))
        self.rows = []
        global_index = 0
        for manifest_path in self.paths:
            root = manifest_path.parent
            records = _stream_rows(manifest_path) if manifest_path.suffix == ".jsonl" else read_rows(manifest_path)
            for raw in records:
                idx = global_index
                global_index += 1
                if self.rank is not None and idx % self.world != self.rank:
                    continue
                row = self._normalize(raw, root)
                row["global_index"] = idx
                self.rows.append(row)
        self.train_indices = [i for i, row in enumerate(self.rows) if row["split"] == "train"]
        if not self.train_indices:
            raise ValueError("The tail manifest has no training rows")

    def _normalize(self, row, root):
        split = row.get("split", "train")
        if split not in ("train", "val", "test"):
            raise ValueError("Split must be train, val, or test")
        uid = str(row.get("uid") or "").strip()
        if not uid:
            raise ValueError("Every tail record needs a uid")
        target_image = row.get("target_image") or row.get("image")
        if not target_image or not _resolve(root, target_image).exists():
            raise ValueError("Tail target image is missing: " + uid)
        state_path = row.get("tail_state_path") or row.get("q_k_path")
        resolved_state = str(_resolve(root, state_path)) if state_path else None
        if resolved_state and not Path(resolved_state).exists():
            raise ValueError("Tail state tensor is missing: " + uid)
        source_image = row.get("source_image")
        resolved_source = str(_resolve(root, source_image)) if source_image else None
        if resolved_source and not Path(resolved_source).exists():
            raise ValueError("Tail source image is missing: " + uid)
        edit = compact(row.get("edit_instruction") or row.get("edit_condition") or row.get("instruction"))
        tail_prompt = compact(row.get("tail_condition") or row.get("target_caption") or row.get("prompt"))
        semantic_task, count_direction = classify_semantic_task(edit, row.get("task"))
        return dict(
            uid=uid,
            split=split,
            target_image=str(_resolve(root, target_image)),
            tail_state_path=resolved_state,
            source_image=resolved_source,
            edit_instruction=edit,
            tail_condition=tail_prompt,
            geometry=row.get("geometry"),
            task=row.get("task"),
            semantic_task=row.get("semantic_task") or semantic_task,
            count_direction=row.get("count_direction") or count_direction,
            is_pure_t2i=bool(row.get("is_pure_t2i", resolved_state is None and not (resolved_source and edit))),
        )

    @property
    def count(self):
        return len(self.train_indices)

    def sample(self, index):
        row = self.rows[index]
        declared = row.get("geometry") or {}
        if row["source_image"] is not None:
            source, target, valid, _ = load_pair(row, self.rt.width, self.rt.height, self.rt.device)
        else:
            crop = declared.get("target") if isinstance(declared, dict) else None
            target, valid, _ = load_canvas(
                row["target_image"], self.rt.width, self.rt.height, self.rt.device,
                crop=None if crop in (None, "full") else crop,
            )
            source = None
        tail_state = None
        if row["tail_state_path"] is not None:
            tail_state = torch.load(row["tail_state_path"], map_location=self.rt.device, weights_only=True)
            if tail_state.ndim == 3:
                tail_state = tail_state.unsqueeze(0)
            tail_state = tail_state.to(device=self.rt.device, dtype=torch.float32)
        return dict(
            uid=row["uid"],
            source_kind="tail_state" if (tail_state is not None or not row["is_pure_t2i"]) else "pure_t2i",
            target=target,
            source=source,
            tail_state=tail_state,
            valid=valid,
            edit_request=row["edit_instruction"],
            tail_condition=row["tail_condition"],
            task=row["task"],
            semantic_task=row["semantic_task"],
            count_direction=row["count_direction"],
            is_pure_t2i=row["is_pure_t2i"],
            role="L_tail",
            keep=False,
        )


class TailSampler:
    """Traverse tail-generator records once per epoch in a shuffled order."""

    def __init__(self, manifest, cfg, rank=0, world=1):
        self.manifest = manifest
        self.cfg = cfg
        self.rank = int(rank)
        self.world = max(1, int(world))
        self.rng = random.Random(int(cfg["seed"]) + int(rank))
        self.epoch_rng = random.Random(int(cfg["seed"]) * 7 + 1 + int(rank))
        self.draws = 0
        if cfg.get("task_weights"):
            raise ValueError(
                "task_weights is retired: traversal is record-equal and shuffled."
            )
        self.indices = list(manifest.train_indices)
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
        return {"L_tail": 1.0}

    @property
    def actual_weights(self):
        return {"L_tail": 1.0}

    def input_domains(self):
        return sorted(self.seen)

    def draw(self):
        if self.cursor >= len(self.order):
            self.epoch += 1
            self._shuffle()
        index = self.order[self.cursor]
        self.cursor += 1
        sample = self.manifest.sample(index)
        self.seen.add(sample["source_kind"])
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
