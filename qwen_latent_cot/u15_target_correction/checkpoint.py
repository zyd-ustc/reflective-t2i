"""Stage checkpoints: strict same-stage resume and weight-only stage handoff."""

import json
import os
import random
import shutil
from pathlib import Path

import torch

from . import CURRENT_GEOMETRY_PROTOCOL, GEOMETRY_PREPROCESSING, GEOMETRY_PROTOCOL, KIND
from . import distributed as dist


def canonical_parameters(module):
    return {
        name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): parameter
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _parameter_layout(cfg, component="editor"):
    if component == "tail":
        param = str(cfg.get("tail_parameterization", "gen_lora"))
        return "tail_gen_full_all_layers" if param == "gen_full" else "tail_gen_lora_all_layers"
    param = str(cfg.get("parameterization", "gen_lora"))
    return "gen_full_all_layers+edit_s_embedder" if param == "gen_full" else "gen_lora_all_layers+edit_s_embedder"


def editor_identity(rt, cfg, component="editor"):
    """The frozen interface every checkpoint must match before weights are loaded.

    ``geometry`` records the canvas preprocessing the weights were trained on,
    so a checkpoint is never loaded under a different wide-image layout.
    """
    condition_protocol = "tail_t2i_v1" if component == "tail" else cfg["condition_protocol"]
    return dict(
        native=rt.identity,
        interface=KIND,
        parameter_layout=_parameter_layout(cfg, component=component),
        condition_protocol=condition_protocol,
        geometry=GEOMETRY_PROTOCOL,
    )

def _assert_writable(path, attempts=30, delay=1.0):
    """Wait for rank 0's directory to become visible on this client.

    The volume is NFSv3: a mkdir issued by rank 0 is not immediately visible to
    the other 63 clients, so a rank that starts writing too early fails with
    "parent directory does not exist". Retry before calling it a storage fault.
    """
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
    path, editor, optimizer, sampler, step, cfg, identity, device,
    *, data_version, input_domains, total_steps, budget, sharded_optimizer=False,
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
        # Replicated training: weights are identical on all ranks.
        # Optimizer state is also saved per-rank when ZeroRedundancyOptimizer is active.
        payload = dict(sampler=sampler.state_dict(), rng=rng(device))
        if sharded_optimizer:
            payload["optimizer"] = cpu(optimizer.state_dict())
    torch.save(payload, partial)
    partial.replace(partial.with_suffix(".pt"))
    dist.barrier()
    if dist.rank() == 0:
        meta = dict(
            kind=KIND,
            stage=int(cfg["stage"]),
            component=str(cfg.get("component", "editor")),
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


def _copy_weights(path, editor, *, rank_file):
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


def resume(path, editor, optimizer, sampler, cfg, identity, *, data_version, total_steps):
    """Resume the same stage; cross-stage loading must use load_stage_weights."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    if int(meta.get("stage", -1)) != int(cfg["stage"]):
        raise ValueError("Strict resume cannot cross stages")
    if int(meta.get("world", -1)) != dist.world_size():
        raise ValueError("Checkpoint world size differs")
    if meta.get("config") != cfg or meta.get("identity") != identity:
        raise ValueError("Checkpoint config or identity differs")
    if meta.get("data_version") != data_version:
        raise ValueError("Checkpoint data version differs")
    geometry_gate(meta.get("identity", {}), allow_legacy=None)
    if int(meta.get("total_steps", -1)) != int(total_steps):
        raise ValueError("Checkpoint total-step budget differs")
    # Weights live in rank 0. Each rank restores its own RNG/sampler state
    # (and its own optimizer shard when sharded_optimizer is enabled).
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


def legacy_geometry_allowed():
    """Operator opt-in for reading a checkpoint trained on the v1 preprocessing.

    Set ``U15_ALLOW_LEGACY_GEOMETRY=1`` only to inspect or compare an old
    checkpoint. Training on new data must start from the v2 preprocessing.
    """
    return os.environ.get("U15_ALLOW_LEGACY_GEOMETRY", "").strip().lower() in ("1", "true", "yes")


def geometry_gate(stored, allow_legacy=None):
    """Refuse to load a checkpoint trained on a different canvas preprocessing.

    The fallback is the v1 identity when the checkpoint predates geometry
    recording, because those runs all used the v1 preprocessing.
    """
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
        f"explicit legacy-geometry opt-in to continue this checkpoint."
    )


def load_stage_weights(path, editor, identity, allow_legacy_geometry=None):
    """Load only trainable weights for Stage 2 or sampling."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    stored = meta.get("identity", {})
    recorded = geometry_gate(stored, allow_legacy_geometry)
    for key in ("native", "interface", "parameter_layout", "condition_protocol", "geometry"):
        if key == "geometry" and recorded != identity.get("geometry"):
            continue
        if stored.get(key) != identity.get(key):
            raise ValueError("Checkpoint identity differs at " + key)
    _copy_weights(path, editor, rank_file="rank_00000.pt")
    return meta


def load_init_weights(path, module, identity, allow_legacy_geometry=None):
    """Warm-start weights into a new stage, including merging LoRA into gen_full."""
    meta = _manifest(path)
    if meta.get("kind") != KIND:
        raise ValueError("Checkpoint kind is not isochronic state flow")
    stored = meta.get("identity", {})
    geometry_gate(stored, allow_legacy_geometry)
    for key in ("native", "interface", "condition_protocol", "geometry"):
        if stored.get(key) != identity.get(key):
            raise ValueError("Checkpoint identity differs at " + key)
    src_layout = stored.get("parameter_layout")
    dst_layout = identity.get("parameter_layout")
    if src_layout == dst_layout:
        _copy_weights(path, module, rank_file="rank_00000.pt")
        return meta
    merge_pairs = {
        ("gen_lora_all_layers+edit_s_embedder", "gen_full_all_layers+edit_s_embedder"),
        ("tail_gen_lora_all_layers", "tail_gen_full_all_layers"),
    }
    if (src_layout, dst_layout) not in merge_pairs:
        raise ValueError(f"Cannot convert checkpoint layout {src_layout} -> {dst_layout}")
    state = torch.load(
        str(Path(path) / "rank_00000.pt"),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    module.merge_lora_weights_(state["weights"])
    return meta


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
