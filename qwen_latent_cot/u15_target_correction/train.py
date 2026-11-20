"""Training loop for the unified isochronic manifold-transition operator."""

import json
import math
import os
import random
import time
from pathlib import Path

import torch

from . import distributed as dist
from .checkpoint import (
    canonical_parameters,
    editor_identity,
    load_init_weights,
    prune_checkpoints,
    resume,
    save,
)
from .config import stage_updates
from .data import LinearManifest, TrainingSampler, data_version as hash_data
from .loading import load_native
from .loss import clean_target_mse
from .model import build_editor


def peak_memory_gib(device):
    module = getattr(torch, device.type, None)
    if module is None or not hasattr(module, "max_memory_allocated"):
        return 0.0
    return float(module.max_memory_allocated(device)) / (1024**3)


def _start_wandb(args, cfg, out, version, budget, steps):
    """Rank-0-only tracking. Offline is the safe default on compute nodes."""
    if dist.rank() != 0:
        return None
    mode = os.environ.get("WANDB_MODE", "offline").strip().lower()
    if mode in ("disabled", "off", "none", "false"):
        return None
    try:
        import wandb
    except Exception:
        return None
    project = os.environ.get("WANDB_PROJECT", "isochronic-manifold-flow")
    entity = os.environ.get("WANDB_ENTITY") or None
    name = os.environ.get("WANDB_RUN_NAME") or f"{Path(args.output).name}-isochronic"
    directory = os.environ.get("WANDB_DIR") or str(out / "wandb")
    tags = [item for item in os.environ.get("WANDB_TAGS", "").split(",") if item]
    try:
        return wandb.init(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=directory,
            tags=tags or None,
            config=dict(
                cfg,
                data_version=version,
                linear=str(args.linear),
                init_checkpoint=getattr(args, "init_checkpoint", None),
                budget=budget,
                steps=steps,
                world=dist.world_size(),
                global_batch=dist.world_size(),
                configured_global_batch=int(cfg["global_batch"]),
            ),
        )
    except Exception as error:
        print("wandb init failed; continuing without tracking: " + str(error), flush=True)
        return None


def learning_rate(step, cfg, total):
    low, high = float(cfg["lr_final"]), float(cfg["lr"])
    warmup = int(cfg["warmup_steps"])
    if warmup > 0 and step <= warmup:
        return low + (high - low) * step / warmup
    if total <= warmup:
        return low
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return low + (high - low) * 0.5 * (1 + math.cos(math.pi * progress))


def isochronic_manifold_update(editor, sample, cfg, rng=None):
    """Short-horizon mid-band noisy-to-clean bridge update on tau in [read_time, train_max_time]."""
    rng = rng or random
    source = sample["source"].float()
    target = sample["target"].float()
    valid = sample["valid"]
    t0 = float(cfg["read_time"])
    t_max = float(cfg.get("train_max_time", cfg["stop_time"]))

    eps = torch.randn_like(source) * editor.rt.noise_scale()
    anchor = t0 * source + (1.0 - t0) * eps

    if rng.random() < float(cfg["bridge_start_prob"]):
        tau = t0
    else:
        u = rng.random()
        tau = t0 + (t_max - t0) * u

    alpha = (tau - t0) / (1.0 - t0)
    state = (1.0 - alpha) * anchor + alpha * target

    dropout = float(cfg.get("edit_prompt_dropout", 0.0))
    prompt_dropped = dropout > 0.0 and rng.random() < dropout
    prefix = (
        editor.empty_prefix()
        if prompt_dropped
        else editor.feedback_prefix(sample["edit_request"])
    )
    predicted_clean = editor.predict_clean(state, tau, prefix)
    loss = clean_target_mse(predicted_clean, target, valid)

    return loss, dict(
        mode="isochronic_manifold",
        role=sample["role"],
        time=float(tau),
        progress=float(alpha),
        start_time=float(t0),
        train_max_time=float(t_max),
        prompt_dropped=int(prompt_dropped),
        clean_calls=1,
    )


def _build_optimizer(params, cfg):
    use_zero = bool(cfg.get("zero_optimizer", False)) and dist.world_size() > 1
    kwargs = dict(
        lr=float(cfg["lr"]),
        betas=tuple(float(v) for v in cfg["betas"]),
        eps=float(cfg["eps"]),
        weight_decay=float(cfg["weight_decay"]),
        foreach=False,
    )
    if use_zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        return ZeroRedundancyOptimizer(list(params.values()), optimizer_class=torch.optim.AdamW, **kwargs), True
    return torch.optim.AdamW(params.values(), **kwargs), False


def _rank_manifest(linear, rank, world):
    """Prefer a pre-split shard so a rank never streams the whole manifest."""
    paths = linear if isinstance(linear, (list, tuple)) else [linear]
    shard_dir = Path(paths[0]).parent / f"shards{world}"
    shard = shard_dir / f"shard_{rank:05d}.jsonl"
    if len(paths) == 1 and shard.exists():
        return [str(shard)], True
    return linear, False


def _version_paths(linear, world):
    """Hash the whole dataset (or its shard manifest), not one rank's slice."""
    paths = linear if isinstance(linear, (list, tuple)) else [linear]
    first = Path(paths[0])
    shard_dir = first.parent / f"shards{world}"
    index = shard_dir / "shards.json"
    if index.exists():
        return [index]
    return [str(p) for p in paths]


def _global_train_records(linear, world, pre_split, manifest):
    """Total training rows across every rank, identical on every rank."""
    if not pre_split:
        if world <= 1 or getattr(manifest, "rank", None) is None:
            return manifest.count
        if dist.world_size() > 1:
            return dist.sum_int(manifest.count, getattr(getattr(manifest, "rt", None), "device", None))
        return manifest.count * world
    index = Path(linear if isinstance(linear, str) else linear[0]).parent / f"shards{world}" / "shards.json"
    if index.exists():
        shards = json.loads(index.read_text())
        return sum(int(entry.get("splits", {}).get("train", 0)) for entry in shards.values())
    return manifest.count * world


def train(args, cfg, device):
    if getattr(args, "resume", None) and getattr(args, "init_checkpoint", None):
        raise ValueError("Cannot pass both --resume and --init-checkpoint")

    out = Path(args.output).resolve()
    if dist.rank() == 0:
        out.mkdir(parents=True, exist_ok=bool(args.resume))
    dist.barrier()

    rt = load_native(args.model, args.source, cfg, device, args.distributed)
    world = dist.world_size()
    if world != int(cfg["global_batch"]):
        raise ValueError(
            f"isochronic_manifold_v1 requires world_size ({world}) == global_batch ({cfg['global_batch']})"
        )

    manifest_paths, pre_split = _rank_manifest(args.linear, dist.rank(), world)
    shard_rank = None if (pre_split or world == 1) else dist.rank()
    manifest = LinearManifest(manifest_paths, rt, cfg, rank=shard_rank, world=world)
    sampler = TrainingSampler(manifest, cfg, dist.rank(), world=world)

    total_records = _global_train_records(args.linear, world, pre_split, manifest)
    budget = stage_updates(total_records, cfg, global_batch=world)
    steps = int(args.steps) if args.steps else budget
    if steps < 1:
        raise ValueError("Training requires at least one update")

    torch.manual_seed(int(cfg["seed"]) + dist.rank())
    train_rng = random.Random(int(cfg["seed"]) + 100003 * (dist.rank() + 1))

    editor = build_editor(rt, cfg).train()
    identity = editor_identity(rt, cfg)
    if getattr(args, "init_checkpoint", None):
        load_init_weights(args.init_checkpoint, editor, identity)

    dist.broadcast_parameters(editor)
    params = canonical_parameters(editor)
    optimizer, sharded_optimizer = _build_optimizer(params, cfg)
    version_paths = _version_paths(args.linear, world) if pre_split else manifest.paths
    version = hash_data(*version_paths)

    start = 0
    if args.resume:
        meta, _ = resume(
            args.resume, editor, optimizer, sampler, cfg, identity,
            data_version=version, total_steps=steps,
        )
        start = int(meta["step"])
        train_rng = random.Random(int(cfg["seed"]) + 100003 * (dist.rank() + 1) + start * 997)

    log = out / "train.jsonl"
    run = _start_wandb(args, cfg, out, version, budget, steps)
    for step in range(start + 1, steps + 1):
        began = time.perf_counter()
        rate = learning_rate(step, cfg, steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        sample = sampler.draw()
        loss, fields = isochronic_manifold_update(editor, sample, cfg, train_rng)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite isochronic manifold loss")
        scaled = loss / dist.world_size()
        scaled.backward()
        dist.sum_gradients(editor, allow_unused=False)
        gradient = torch.nn.utils.clip_grad_norm_(list(params.values()), float(cfg["gradient_clip"]))
        if not torch.isfinite(gradient):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        stats = dict(
            step=step,
            lr=rate,
            loss=float(loss.detach()),
            gradient_norm=float(gradient),
            clipped=float(gradient) > float(cfg["gradient_clip"]),
            seconds=time.perf_counter() - began,
            source_kind=sample["source_kind"],
            role=fields["role"],
            task=sample.get("task"),
            semantic_task=sample.get("semantic_task"),
            count_direction=sample.get("count_direction"),
            mode=fields["mode"],
            uid=sample["uid"],
            prompt_dropped=fields["prompt_dropped"],
            time=fields["time"],
            progress=fields["progress"],
            start_time=fields["start_time"],
            train_max_time=fields["train_max_time"],
            clean_calls=fields["clean_calls"],
            input_domains=sampler.input_domains(),
            peak_memory_gib=peak_memory_gib(device),
            world=dist.world_size(),
            global_batch=dist.world_size(),
        )
        if dist.rank() == 0:
            with log.open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
            if run is not None:
                run.log(stats, step=step)
        if step % int(cfg["save_every"]) == 0 or step == steps:
            save(
                out / f"step_{step:06d}",
                editor,
                optimizer,
                sampler,
                step,
                cfg,
                identity,
                device,
                data_version=version,
                input_domains=sampler.input_domains(),
                total_steps=steps,
                budget=budget,
                sharded_optimizer=sharded_optimizer,
            )
            pruned = prune_checkpoints(out, int(cfg.get("keep_checkpoints", 0)))
            if pruned and dist.rank() == 0:
                print("pruned checkpoints: " + ", ".join(pruned), flush=True)
    if dist.rank() == 0:
        (out / "complete.json").write_text(
            json.dumps(
                dict(
                    stage=1,
                    component="editor",
                    steps=steps,
                    budget=budget,
                    override=steps != budget,
                    input_domains=sampler.input_domains(),
                    data_version=version,
                    world=dist.world_size(),
                    global_batch=dist.world_size(),
                )
            )
            + "\n"
        )
        if run is not None:
            run.finish()
