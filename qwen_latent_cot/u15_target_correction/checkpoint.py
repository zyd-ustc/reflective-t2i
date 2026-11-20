"""Checkpoints for the unified isochronic manifold-transition operator."""

import json
import os
import random
import shutil
from pathlib import Path

import torch

from . import (
    COMPAT_CHECKPOINT_INTERFACES,
    CONDITION_PROTOCOL,
    CURRENT_GEOMETRY_PROTOCOL,
    EDITOR_INTERFACE,
    GEOMETRY_PREPROCESSING,
    GEOMETRY_PROTOCOL,
    KIND,
    PARAMETER_LAYOUT,
)
from . import distributed as dist


def canonical_parameters(module):
    return {
        name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def editor_identity(rt, cfg):
    """Frozen architectural identity recorded in every checkpoint manifest."""
    return dict(
        native=rt.identity,
        interface=KIND,
        editor_interface=EDITOR_INTERFACE,
        parameter_layout=PARAMETER_LAYOUT,
        condition_protocol=CONDITION_PROTOCOL,
        prediction_type="clean_image",
        bridge_type="source_noised_to_target_clean_v1",
        read_time=float(cfg["read_time"]),
        endpoint_time=float(cfg.get("bridge_endpoint_time", 1.0)),
        geometry=GEOMETRY_PROTOCOL,
    )


def _assert_writable(path, attempts=30, delay=1.0):
    """Wait for rank 0's directory to become visible across NFS clients."""
    import time as _time

    for _ in range(max(1, attempts)):
        if path.is_dir():
            return
        _time.sleep(delay)
    raise RuntimeError(
        "Checkpoint directory never became visible (storage full, quota "
        "exhausted, or a stale NFS handle): " + str(path)
    )


def prune_checkpoints(root, keep):
    """Keep the newest ``keep`` step directories; returns the names removed."""
    if not keep or keep < 1 or dist.rank() != 0:
        return []
    steps = sorted(
        (p for p in Path(root).glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    stale = steps[:-keep] if len(steps) > keep else []
    removed = []
    for directory in stale:
        try:
            shutil.rmtree(directory)
            removed.append(directory.name)
        except OSError:
            pass
    return removed


def save(
    path,
    editor,
    optimizer,
    sampler,
    step,
    cfg,
    identity,
    device,
    *,
    data_version,
    input_domains,
    total_steps,
    budget,
    sharded_optimizer=False,
):
    path = Path(path)
    if dist.rank() == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    _assert_writable(path)
    partial = path / f"rank_{dist.rank():05d}.partial"
    if dist.rank() == 0:
        payload = dict(
            weights=cpu(canonical_parameters(editor)),
            optimizer=cpu(optimizer.state_dict()),
            sampler=sampler.state_dict(),
            rng=rng(device),
        )
    else:
        payload = dict(sampler=sampler.state_dict(), rng=rng(device))
        if sharded_optimizer:
            payload["optimizer"] = cpu(optimizer.state_dict())
    torch.save(payload, partial)
    partial.replace(partial.with_suffix(".pt"))
    dist.barrier()
    if dist.rank() == 0:
        meta = dict(
            kind=KIND,
            stage=1,
            component="editor",
            step=int(step),
            world=dist.world_size(),
            config=cfg,
            identity=identity,
            parameter_layout=identity["parameter_layout"],
            sharded_optimizer=bool(sharded_optimizer),
            data_version=data_version,
            input_domains=list(input_domains),
            total_steps=int(total_steps),
            budget=int(budget),
        )
        (path / "manifest.partial").write_text(json.dumps(meta, indent=2) + "\n")
        (path / "manifest.partial").replace(path / "manifest.json")
    dist.barrier()


def _manifest(path):
    return json.loads((Path(path) / "manifest.json").read_text())


def _copy_weights(path, editor, *, rank_file="rank_00000.pt"):
    params = canonical_parameters(editor)
    state = torch.load(
        str(Path(path) / rank_file),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    weights = state["weights"]
    if params.keys() != weights.keys():
        missing = sorted(params.keys() ^ weights.keys())
        raise ValueError("Editor parameter inventory differs: " + ", ".join(missing[:8]))
    with torch.no_grad():
        for name, parameter in params.items():
            if parameter.shape != weights[name].shape:
                raise ValueError("Parameter shape differs: " + name)
            parameter.copy_(weights[name].to(parameter))
    return state


def legacy_geometry_allowed():
    return os.environ.get("U15_ALLOW_LEGACY_GEOMETRY", "").strip().lower() in ("1", "true", "yes")


def geometry_gate(stored, allow_legacy=None):
    """Refuse to load a checkpoint trained on a different canvas preprocessing."""
    if allow_legacy is None:
        allow_legacy = legacy_geometry_allowed()
    recorded = (stored or {}).get("geometry") or "shared_field_of_view_v1"
    if recorded == CURRENT_GEOMETRY_PROTOCOL:
        return recorded
    if allow_legacy:
        return recorded
    preprocessing = GEOMETRY_PREPROCESSING.get(recorded, "unknown preprocessing")
    raise ValueError(
        f"Checkpoint geometry {recorded} predates {CURRENT_GEOMETRY_PROTOCOL} "
        f"({preprocessing}). Retrain on the current preprocessing, or pass an "
        f"explicit legacy-geometry opt-in to inspect this checkpoint."
    )


def _check_compatible_weights_identity(stored, identity, recorded_geometry=None):
    for key in ("native", "interface", "parameter_layout", "condition_protocol", "prediction_type", "bridge_type", "read_time", "endpoint_time", "geometry"):
        if key == "geometry" and recorded_geometry is not None and recorded_geometry != identity.get("geometry"):
            continue
        if stored.get(key) != identity.get(key):
            raise ValueError(f"Checkpoint identity differs at {key}: {stored.get(key)!r} != {identity.get(key)!r}")
    if stored.get("editor_interface") not in COMPAT_CHECKPOINT_INTERFACES:
        raise ValueError(f"Unsupported checkpoint editor_interface: {stored.get('editor_interface')!r}")


def resume(path, editor, optimizer, sampler, cfg, identity, *, data_version, total_steps):
    """Strict same-run resume."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    if int(meta.get("world", -1)) != dist.world_size():
        raise ValueError("Checkpoint world size differs")
    if meta.get("config") != cfg or meta.get("identity") != identity:
        raise ValueError("Checkpoint config or identity differs")
    if meta.get("data_version") != data_version:
        raise ValueError("Checkpoint data version differs")
    geometry_gate(meta.get("identity", {}), allow_legacy=None)
    if int(meta.get("total_steps", -1)) != int(total_steps):
        raise ValueError("Checkpoint total-step budget differs")
    base = _copy_weights(path, editor, rank_file="rank_00000.pt")
    if dist.rank() == 0:
        own = base
    else:
        own = torch.load(
            str(Path(path) / f"rank_{dist.rank():05d}.pt"),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    if meta.get("sharded_optimizer"):
        optimizer.load_state_dict(own["optimizer"])
    else:
        optimizer.load_state_dict(base["optimizer"])
    sampler.load_state_dict(own["sampler"])
    restore_rng(own["rng"], next(editor.parameters()).device)
    return meta, base


def load_stage_weights(path, editor, identity, allow_legacy_geometry=None):
    """Load trained LoRA weights for inference or warm-start."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    stored = meta.get("identity", {})
    recorded = geometry_gate(stored, allow_legacy_geometry)
    _check_compatible_weights_identity(stored, identity, recorded_geometry=recorded)
    _copy_weights(path, editor, rank_file="rank_00000.pt")
    return meta


def load_init_weights(path, editor, identity, allow_legacy_geometry=None):
    """Warm-start weights from a compatible GEN LoRA checkpoint."""
    return load_stage_weights(path, editor, identity, allow_legacy_geometry=allow_legacy_geometry)


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu(item) for item in value)
    return value


def rng(device):
    return dict(
        cpu=torch.get_rng_state(),
        python=random.getstate(),
        device=getattr(torch, device.type).get_rng_state(device).cpu() if device.type != "cpu" else None,
    )


def restore_rng(saved, device):
    torch.set_rng_state(saved["cpu"])
    random.setstate(saved["python"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(saved["device"], device)
