"""Generator-agnostic edit-pair dataset reusing prod_1m_glm53_20261118 + shared_field_of_view_v2."""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset


DEFAULT_PATH_REMAPS: tuple[tuple[str, str], ...] = (
    ("/mnt/sfs/unimoe_dataset", "/data/unimoe_dataset"),
)


def _active_path_remaps(
    path_remaps: Sequence[tuple[str, str]] = DEFAULT_PATH_REMAPS,
) -> tuple[tuple[str, str], ...]:
    """Activate a prefix remap only on hosts where old_prefix is absent and new_prefix exists."""
    active = []
    for old_prefix, new_prefix in path_remaps:
        if not Path(old_prefix).exists() and Path(new_prefix).exists():
            active.append((old_prefix, new_prefix))
    return tuple(active)


def _resolve_path(
    root: Path,
    value: str,
    active_remaps: Sequence[tuple[str, str]] = (),
) -> Path:
    raw = str(value)
    for old_prefix, new_prefix in active_remaps:
        if raw.startswith(old_prefix):
            return Path(new_prefix + raw[len(old_prefix) :])
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def _crop_box(img: Image.Image, box: Sequence[float] | None) -> Image.Image:
    if not box or len(box) != 4:
        return img
    w, h = img.size
    x0, y0, x1, y1 = [float(v) for v in box]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0 + 1e-4:
        x0, x1 = x0 * w, x1 * w
        y0, y1 = y0 * h, y1 * h
    left = max(0, min(w - 1, int(round(x0))))
    top = max(0, min(h - 1, int(round(y0))))
    right = max(left + 1, min(w, int(round(x1))))
    bottom = max(top + 1, min(h, int(round(y1))))
    return img.crop((left, top, right, bottom))


def load_shared_fov_pair(
    source_path: str | Path,
    target_path: str | Path,
    width: int,
    height: int,
    *,
    geometry: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load `(source, target, valid_mask)` in [-1, 1] RGB aligned on a shared letterbox canvas."""
    with Image.open(source_path) as s_im:
        src = s_im.convert("RGB")
    with Image.open(target_path) as t_im:
        tar = t_im.convert("RGB")

    if isinstance(geometry, dict):
        src = _crop_box(src, geometry.get("source_crop"))
        tar = _crop_box(tar, geometry.get("target_crop"))

    sw, sh = src.size
    scale = min(float(width) / max(sw, 1), float(height) / max(sh, 1))
    rw = max(1, min(int(width), int(round(sw * scale))))
    rh = max(1, min(int(height), int(round(sh * scale))))
    x_off = (int(width) - rw) // 2
    y_off = (int(height) - rh) // 2

    src_resized = src.resize((rw, rh), Image.Resampling.BICUBIC)
    tar_resized = tar.resize((rw, rh), Image.Resampling.BICUBIC)

    src_canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
    tar_canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
    src_canvas.paste(src_resized, (x_off, y_off))
    tar_canvas.paste(tar_resized, (x_off, y_off))

    import numpy as np

    src_t = torch.from_numpy(np.asarray(src_canvas, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    tar_t = torch.from_numpy(np.asarray(tar_canvas, dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    valid = torch.zeros((1, int(height), int(width)), dtype=torch.float32)
    valid[:, y_off : y_off + rh, x_off : x_off + rw] = 1.0
    return src_t, tar_t, valid


class ReflectiveEditPairDataset(Dataset):
    """Loads approved edit pairs from one or more JSONL manifests (with optional Pilot subsampling)."""

    def __init__(
        self,
        manifest_paths: str | Sequence[str | Path],
        *,
        width: int,
        height: int,
        split: str = "train",
        max_records: int | None = None,
        seed: int = 31,
        verify_files: bool = False,
        path_remaps: Sequence[tuple[str, str]] = DEFAULT_PATH_REMAPS,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.split = str(split)
        self.path_remaps = _active_path_remaps(path_remaps)

        if isinstance(manifest_paths, (str, Path)):
            expanded = sorted(glob.glob(str(manifest_paths)))
            paths = [Path(p).resolve() for p in (expanded if expanded else [str(manifest_paths)])]
        else:
            paths = []
            for item in manifest_paths:
                expanded = sorted(glob.glob(str(item)))
                paths.extend(Path(p).resolve() for p in (expanded if expanded else [str(item)]))

        rows: list[dict[str, Any]] = []
        family_splits: dict[str, str] = {}
        for mpath in paths:
            root = mpath.parent
            with mpath.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    raw = json.loads(line)
                    row_split = str(raw.get("split", "train"))
                    fam = str(
                        raw.get("source_family")
                        or hashlib.sha1(str(raw.get("source_image", "")).encode()).hexdigest()[:16]
                    )
                    prev_split = family_splits.setdefault(fam, row_split)
                    if prev_split != row_split:
                        raise ValueError(f"source_family {fam} crosses splits ({prev_split} vs {row_split})")
                    if row_split != self.split:
                        continue
                    if not bool(raw.get("semantic_approved", raw.get("approved", True))):
                        continue
                    src = raw.get("source_image")
                    tar = raw.get("target_image")
                    edit = " ".join(
                        str(raw.get("edit_instruction") or raw.get("edit_condition") or raw.get("instruction") or "").split()
                    )
                    if not src or not tar or not edit:
                        continue
                    src_p = _resolve_path(root, src, self.path_remaps)
                    tar_p = _resolve_path(root, tar, self.path_remaps)
                    if verify_files and (not src_p.exists() or not tar_p.exists()):
                        continue
                    action_tag = (
                        raw.get("edit_semantics", {}).get("edit_action")
                        if isinstance(raw.get("edit_semantics"), dict)
                        else None
                    ) or raw.get("verified_task_type") or raw.get("semantic_task") or raw.get("task") or "other"
                    rows.append(
                        {
                            "uid": str(raw.get("uid") or f"{mpath.stem}_{len(rows)}"),
                            "source_image": str(src_p),
                            "target_image": str(tar_p),
                            "edit_instruction": edit,
                            "original_prompt": str(raw.get("source_caption") or raw.get("original_prompt") or "").strip(),
                            "target_caption": str(raw.get("target_caption") or raw.get("tail_condition") or "").strip(),
                            "geometry": raw.get("geometry"),
                            "semantic_task": str(action_tag),
                        }
                    )

        if max_records is not None and len(rows) > int(max_records):
            rng = random.Random(int(seed))
            rng.shuffle(rows)
            rows = rows[: int(max_records)]

        if not rows:
            raise ValueError(f"No valid rows found for split={self.split} in {paths}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        src, tar, valid = load_shared_fov_pair(
            row["source_image"],
            row["target_image"],
            self.width,
            self.height,
            geometry=row["geometry"],
        )
        return {
            "uid": row["uid"],
            "source_rgb": src,
            "target_rgb": tar,
            "valid": valid,
            "edit_instruction": row["edit_instruction"],
            "original_prompt": row["original_prompt"],
            "target_caption": row["target_caption"],
            "semantic_task": row["semantic_task"],
        }
