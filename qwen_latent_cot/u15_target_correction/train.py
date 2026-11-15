"""Training loop for the isochronic state editor (IFM + rollout + CFG) and tail generator."""

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
    load_stage_weights,
    prune_checkpoints,
    resume,
    save,
)
from .config import stage_updates
from .data import LinearManifest, TailManifest, TailSampler, TrainingSampler, data_version as hash_data
from .field import edit_state, phase_grid, rollout_prefix
from .loading import load_native
from .loss import endpoint_flow_matching, state_flow_matching, tail_flow_matching
from .model import StateEditor, TailGenerator


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
    project = os.environ.get("WANDB_PROJECT", "isochronic-state-flow")
    entity = os.environ.get("WANDB_ENTITY") or None
    component = str(cfg.get("component", "editor"))
    name = os.environ.get("WANDB_RUN_NAME") or f"{Path(args.output).name}-{component}-stage{cfg['stage']}"
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


def _linear_fields(sample, editor, cfg, grid):
    source, target = sample["source"], sample["target"]
    sigma = editor.rt.noise_scale()
    noise = torch.randn(source.shape, device=source.device, dtype=source.dtype)
    time = sample["time"]
    anchor = time * source + (1 - time) * sigma * noise
    destination = time * target + (1 - time) * sigma * noise
    # No lookahead: the editor reads a_t and the instruction, nothing else.
    return anchor, destination, None, sample["valid"], 0


def _field_update(editor, anchor, destination, time, joint, base, valid):
    """One standard IFM sample: predict the paired displacement at q_s."""
    progress = float(torch.rand((), device=anchor.device))
    state = (1 - progress) * anchor + progress * destination
    predicted = editor.state_velocity(state, time, progress, joint, base)
    return state_flow_matching(predicted, destination - anchor, time, valid), progress


def _rollout_step_update(editor, anchor, destination, time, prefixes, active_prefix,
                         base, valid, cfg, rng, prompt_dropped):
    """Detached k-step rollout to q_k, then remaining-endpoint supervision."""
    steps = int(cfg["edit_steps"])
    step_index = rng.randrange(steps)
    progress = step_index / steps
    rollout_cfg_prob = float(cfg.get("rollout_cfg_prob", 0.0))
    cfg_scale = float(cfg.get("rollout_cfg_scale", 1.0))
    use_cfg = (
        not prompt_dropped
        and step_index > 0
        and rollout_cfg_prob > 0.0
        and cfg_scale > 1.0
        and rng.random() < rollout_cfg_prob
    )
    state, _ = rollout_prefix(
        editor, anchor, time, prefixes, step_index, steps,
        base_prediction=base,
        cfg_scale=cfg_scale if use_cfg else 1.0,
        joint_prefix=active_prefix,
    )
    predicted = editor.state_velocity(state, time, progress, active_prefix, base)
    predicted_endpoint = state + (1.0 - progress) * predicted
    loss = endpoint_flow_matching(predicted_endpoint, destination, time, valid)
    state_calls = 1 + step_index * (2 if use_cfg else 1)
    return loss, progress, step_index, use_cfg, state_calls


def _rollout_terminal_update(editor, anchor, destination, time, prefixes, active_prefix,
                             base, valid, cfg):
    """Full K-step differentiable rollout supervising the final edited state q_K."""
    steps = int(cfg["edit_steps"])
    edited, _ = edit_state(
        editor, anchor, time, prefixes, steps,
        base_prediction=base, grad=True, cfg_scale=1.0, joint_prefix=active_prefix,
    )
    loss = endpoint_flow_matching(edited, destination, time, valid)
    return loss, 1.0, steps, steps


def _keep_update(editor, anchor, time, active_prefix, base, valid):
    """A keep draw has to move nothing at any edit progress above zero."""
    progress = 0.5 + 0.5 * float(torch.rand((), device=anchor.device))
    predicted = editor.state_velocity(anchor, time, progress, active_prefix, base)
    zero = torch.zeros_like(predicted)
    return state_flow_matching(predicted, zero, time, valid), progress


def _update_keep(editor, sample, anchor, destination, time, valid, preview_calls):
    """Supervise the keep instruction as a no-op edit, never as a paired edit."""
    prefixes = editor.prefixes(sample["edit_request"])
    base = editor.image_clean(anchor, time, prefixes["image"])
    loss, progress = _keep_update(editor, anchor, time, prefixes["joint"], base, valid)
    return loss, dict(
        time=time,
        progress=progress,
        mode="keep",
        role=sample["role"],
        preview_calls=preview_calls,
        baseline_calls=1,
        state_calls=1,
        keep=1,
        prompt_dropped=0,
        rollout_step=None,
        rollout_cfg_used=0,
    )


def _update(editor, sample, cfg, grid, rng=None):
    rng = rng or random
    anchor, destination, _, valid, preview_calls = _linear_fields(sample, editor, cfg, grid)
    time = float(sample["time"])
    keep = bool(sample["keep"])
    if keep:
        # target is a clone of the source, so both flow directions vanish here.
        return _update_keep(editor, sample, anchor, destination, time, valid, preview_calls)
    dropout = float(cfg.get("edit_prompt_dropout", 0.0))
    prompt_dropped = (not keep) and dropout > 0.0 and rng.random() < dropout
    prefixes = editor.prefixes(None if prompt_dropped else sample["edit_request"], allow_empty=prompt_dropped)
    active_prefix = prefixes["empty"] if prompt_dropped else prefixes["joint"]
    base = editor.image_clean(anchor, time, prefixes["image"])

    rollout_prob = float(cfg.get("rollout_prob", 0.0))
    terminal_prob = float(cfg.get("terminal_rollout_prob", 0.0))
    draw = rng.random() if (rollout_prob > 0.0 or terminal_prob > 0.0) else 1.0

    rollout_step = None
    rollout_cfg_used = False
    if draw < terminal_prob:
        loss, progress, rollout_step, state_calls = _rollout_terminal_update(
            editor, anchor, destination, time, prefixes, active_prefix, base, valid, cfg
        )
        mode = "rollout_terminal"
    elif draw < terminal_prob + rollout_prob:
        loss, progress, rollout_step, rollout_cfg_used, state_calls = _rollout_step_update(
            editor, anchor, destination, time, prefixes, active_prefix,
            base, valid, cfg, rng, prompt_dropped,
        )
        mode = "rollout_step"
    else:
        loss, progress = _field_update(editor, anchor, destination, time, active_prefix, base, valid)
        state_calls = 1
        mode = "ifm"

    return loss, dict(
        time=time,
        progress=progress,
        mode=mode,
        role=sample["role"],
        preview_calls=preview_calls,
        baseline_calls=1,
        state_calls=state_calls,
        keep=int(keep),
        prompt_dropped=int(prompt_dropped),
        rollout_step=rollout_step,
        rollout_cfg_used=int(rollout_cfg_used),
    )


def _tail_update(tail, sample, cfg, editor, rng):
    """Supervise standalone tail generator on q_K -> x_b and optional native T2I mix."""
    target, valid = sample["target"], sample["valid"]
    t0 = float(cfg.get("tail_t_min", cfg["read_time"]))
    t1 = float(cfg.get("tail_t_max", 1.0))
    t_eps = float(getattr(tail.model.config, "t_eps", cfg["t_eps"]))
    native_prob = float(cfg.get("tail_native_prob", 0.0))
    use_native = bool(sample["is_pure_t2i"]) or (native_prob > 0.0 and rng.random() < native_prob)

    if not use_native:
        q_k = sample["tail_state"]
        if q_k is None:
            if editor is None or sample["source"] is None or not sample["edit_request"]:
                raise ValueError("Non-pure tail sample requires tail_state_path or online editor+source+instruction")
            with torch.no_grad():
                sigma = editor.rt.noise_scale()
                noise = torch.randn(sample["source"].shape, device=sample["source"].device, dtype=sample["source"].dtype)
                anchor = t0 * sample["source"] + (1.0 - t0) * sigma * noise
                epref = editor.prefixes(sample["edit_request"])
                q_k, _ = edit_state(
                    editor, anchor, t0, epref, int(cfg["edit_steps"]),
                    cfg_scale=float(cfg.get("edit_cfg_scale", 1.0)),
                )
        alpha = rng.random() * (1.0 - 1e-4)
        tau = t0 + (t1 - t0) * alpha
        z_tau = (1.0 - alpha) * q_k.float() + alpha * target.float()
        mode = "tail_edited"
    else:
        tau = t0 + (t1 - t0) * (rng.random() * (1.0 - 1e-4))
        sigma = tail.rt.noise_scale()
        noise = torch.randn(target.shape, device=target.device, dtype=target.dtype)
        z_tau = tau * target.float() + (1.0 - tau) * sigma * noise
        mode = "tail_native"

    tail_text = sample["tail_condition"] or None
    drop_prob = float(cfg.get("tail_prompt_dropout", 0.0))
    prompt_dropped = (tail_text is not None) and drop_prob > 0.0 and rng.random() < drop_prob
    raw_prompt = None if prompt_dropped else tail_text
    with torch.no_grad():
        prefix = tail.rt.prefix(raw_prompt)
    prefix_tuple = (prefix.cache, prefix.indexes, None)
    predicted_clean = tail.predict_clean(z_tau, tau, prefix_tuple)
    loss = tail_flow_matching(predicted_clean, z_tau, target, tau, t_eps, valid)

    return loss, dict(
        time=tau,
        progress=0.0,
        mode=mode,
        role=sample["role"],
        preview_calls=0,
        baseline_calls=0,
        state_calls=0,
        tail_calls=1,
        keep=0,
        prompt_dropped=int(prompt_dropped),
        rollout_step=None,
        rollout_cfg_used=0,
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
    component = str(cfg.get("component", "editor"))
    world = dist.world_size()
    # Read only this rank's shard file when the manifest was pre-split;
    # otherwise fall back to streaming and filtering by stride.
    manifest_paths, pre_split = _rank_manifest(args.linear, dist.rank(), world)
    shard_rank = None if (pre_split or world == 1) else dist.rank()
    if component == "tail":
        manifest = TailManifest(
            manifest_paths, rt, cfg, rank=shard_rank, world=world,
        )
        sampler = TailSampler(manifest, cfg, dist.rank(), world=world)
    else:
        captions = getattr(args, "target_captions", None)
        manifest = LinearManifest(
            manifest_paths, rt, cfg, captions=captions,
            rank=shard_rank, world=world,
        )
        sampler = TrainingSampler(manifest, cfg, dist.rank(), world=world)

    # The budget counts traversals, not a leftover step cap. One step consumes
    # one record per rank, so one pass over N records takes N / world steps.
    # Per-shard sizes differ by a few rows, so the total must not be derived from
    # this rank's slice: resume compares total_steps across ranks and needs one
    # shared value. Shard counts are summed from the split index instead.
    total_records = _global_train_records(args.linear, world, pre_split, manifest)
    budget = stage_updates(total_records, cfg, global_batch=world)
    steps = int(args.steps) if args.steps else budget
    if steps < 1:
        raise ValueError("The stage requires at least one update")

    grid = phase_grid(rt, cfg)
    torch.manual_seed(int(cfg["seed"]) + dist.rank())
    train_rng = random.Random(int(cfg["seed"]) + 100003 * (dist.rank() + 1))

    frozen_editor = None
    if component == "tail":
        module = TailGenerator(rt, cfg).train()
        editor_ckpt = getattr(args, "tail_editor_checkpoint", None)
        if editor_ckpt:
            editor_cfg = dict(cfg)
            if getattr(args, "tail_editor_config", None):
                from .config import load_config
                editor_cfg = load_config(args.tail_editor_config)
            frozen_editor = StateEditor(rt, editor_cfg).eval().requires_grad_(False)
            load_stage_weights(editor_ckpt, frozen_editor, editor_identity(rt, editor_cfg, component="editor"))
    else:
        module = StateEditor(rt, cfg).train()

    identity = editor_identity(rt, cfg, component=component)
    if getattr(args, "init_checkpoint", None):
        load_init_weights(args.init_checkpoint, module, identity)

    dist.broadcast_parameters(module)
    params = canonical_parameters(module)
    optimizer, sharded_optimizer = _build_optimizer(params, cfg)
    # The version must describe the dataset, not this rank's slice: hashing a
    # shard file would give every rank a different value and break strict resume.
    version_paths = _version_paths(args.linear, world) if pre_split else manifest.paths
    version = hash_data(*version_paths, *manifest.captions_paths)

    start = 0
    if args.resume:
        meta, _ = resume(
            args.resume, module, optimizer, sampler, cfg, identity,
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
        if component == "tail":
            loss, fields = _tail_update(module, sample, cfg, frozen_editor, train_rng)
        else:
            loss, fields = _update(module, sample, cfg, grid, train_rng)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite state-flow loss")
        scaled = loss / dist.world_size()
        scaled.backward()
        dist.sum_gradients(module, allow_unused=False)
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
            keep=fields["keep"],
            prompt_dropped=fields["prompt_dropped"],
            rollout_step=fields["rollout_step"],
            rollout_cfg_used=fields["rollout_cfg_used"],
            time=fields["time"],
            progress=fields["progress"],
            preview_calls=fields["preview_calls"],
            baseline_calls=fields["baseline_calls"],
            state_calls=fields["state_calls"],
            input_domains=sampler.input_domains(),
            configured_weights=sampler.configured_weights,
            actual_weights=sampler.actual_weights,
            peak_memory_gib=peak_memory_gib(device),
            world=dist.world_size(),
            global_batch=dist.world_size(),
            configured_global_batch=int(cfg["global_batch"]),
            batch_mismatch=dist.world_size() != int(cfg["global_batch"]),
        )
        for key in ("structure", "perceptual", "preserve", "frequency", "hf_ratio", "edited_fraction", "tail_calls"):
            if key in fields:
                stats[key] = fields[key]
        if dist.rank() == 0:
            with log.open("a") as handle:
                handle.write(json.dumps(stats) + "\n")
            if run is not None:
                run.log(dict(stats, stage=int(cfg["stage"])), step=step)
        if step % int(cfg["save_every"]) == 0 or step == steps:
            save(
                out / f"step_{step:06d}",
                module,
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
            # The shared volume is near its ceiling; keep a bounded window.
            pruned = prune_checkpoints(out, int(cfg.get("keep_checkpoints", 0)))
            if pruned and dist.rank() == 0:
                print("pruned checkpoints: " + ", ".join(pruned), flush=True)
    if dist.rank() == 0:
        (out / "complete.json").write_text(
            json.dumps(
                dict(
                    stage=int(cfg["stage"]),
                    component=component,
                    steps=steps,
                    budget=budget,
                    override=steps != budget,
                    input_domains=sampler.input_domains(),
                    configured_weights=sampler.configured_weights,
                    actual_weights=sampler.actual_weights,
                    data_version=version,
                    world=dist.world_size(),
                    global_batch=dist.world_size(),
                    configured_global_batch=int(cfg["global_batch"]),
                    batch_mismatch=dist.world_size() != int(cfg["global_batch"]),
                )
            )
            + "\n"
        )
        if run is not None:
            run.finish()
