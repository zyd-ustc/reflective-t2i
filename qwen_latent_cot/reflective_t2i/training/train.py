"""Generator-agnostic State Corrector training step & production DDP loop for reflective_t2i_v1."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

import torch
import torch.distributed as dist

from ..core.contracts import ConditionMode
from .losses import LossConfig, ReflectiveCorrectionLoss
from .rollout import RolloutConfig, build_training_batch_step


@dataclass
class TrainConfig:
    lr: float = 3e-5
    lr_final: float = 5e-6
    warmup_steps: int = 200
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    gradient_clip: float = 1.0
    grad_accum: int = 1
    condition_mode: ConditionMode = "native_prompt_plus_feedback"
    save_every: int = 500
    keep_checkpoints: int = 4
    log_every: int = 10
    seed: int = 31
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    loss: LossConfig = field(default_factory=LossConfig)


def init_distributed(device_arg: str = "npu:0") -> tuple[str, int, int, int]:
    """Initialize HCCL (NPU) or NCCL (CUDA) DDP when launched under ``torchrun``."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if device_arg.startswith("npu"):
        import torch_npu  # noqa: F401
        dev = f"npu:{local_rank}" if world > 1 else device_arg
        torch.npu.set_device(dev)
        backend = "hccl"
    elif device_arg.startswith("cuda"):
        dev = f"cuda:{local_rank}" if world > 1 else device_arg
        torch.cuda.set_device(dev)
        backend = "nccl"
    else:
        dev = device_arg
        backend = "gloo"

    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return dev, rank, local_rank, world


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _broadcast_trainable(params: list[torch.nn.Parameter]) -> None:
    if dist.is_initialized() and dist.get_world_size() > 1:
        for p in params:
            dist.broadcast(p.data, src=0)


def _sync_gradients(params: list[torch.nn.Parameter]) -> None:
    if not (dist.is_initialized() and dist.get_world_size() > 1):
        return
    world = float(dist.get_world_size())
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world)


def cosine_lr(step: int, total_steps: int, cfg: TrainConfig) -> float:
    lo, hi = float(cfg.lr_final), float(cfg.lr)
    warmup = int(cfg.warmup_steps)
    if warmup > 0 and step <= warmup:
        return lo + (hi - lo) * step / warmup
    if total_steps <= warmup:
        return lo
    prog = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
    return lo + (hi - lo) * 0.5 * (1.0 + math.cos(math.pi * prog))


def training_step(
    generator_adapter,
    sample: dict[str, Any],
    loss_fn: ReflectiveCorrectionLoss,
    train_cfg: TrainConfig,
    *,
    step: int = 1,
    null_cond=None,
    rng: random.Random | None = None,
    torch_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one generator-agnostic forward + loss step for any GeneratorAdapter."""
    dev = generator_adapter.device
    src_rgb = sample["source_rgb"]
    tar_rgb = sample["target_rgb"]
    valid = sample.get("valid")
    if src_rgb.ndim == 3:
        src_rgb = src_rgb.unsqueeze(0)
        tar_rgb = tar_rgb.unsqueeze(0)
        if valid is not None and valid.ndim == 3:
            valid = valid.unsqueeze(0)

    with torch.no_grad():
        source_clean = generator_adapter.encode_clean(src_rgb.to(dev))
        target_clean = generator_adapter.encode_clean(tar_rgb.to(dev))
        edit_cond = generator_adapter.encode_edit_condition(
            feedback=sample["edit_instruction"],
            original_prompt=sample.get("original_prompt") or None,
            target_rewrite=sample.get("target_caption") or None,
            condition_mode=train_cfg.condition_mode,
        )
        if null_cond is None:
            null_cond = generator_adapter.encode_prompt(None)

    state, progress, active_cond, sup_target, r_stats = build_training_batch_step(
        generator_adapter,
        source_clean,
        target_clean,
        edit_cond,
        null_cond,
        train_cfg.rollout,
        step=step,
        rng=rng,
        torch_generator=torch_generator,
    )

    predicted_clean = generator_adapter.predict_clean(state, progress, active_cond, use_corrector=True)
    loss, l_stats = loss_fn(predicted_clean, sup_target, source_clean, valid)

    metrics = {
        "loss": float(loss.detach().item()),
        **r_stats,
        **l_stats,
        "uid": sample.get("uid", ""),
        "semantic_task": sample.get("semantic_task", ""),
    }
    return loss, metrics


def save_corrector_checkpoint(
    out_dir: str | Path,
    generator_adapter,
    step: int,
    config_payload: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    trainer_state: dict[str, Any] | None = None,
    keep_checkpoints: int = 0,
) -> Path:
    out_path = Path(out_dir)
    ckpt_dir = out_path / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if generator_adapter.corrector is None:
        raise RuntimeError("GeneratorAdapter has no attached corrector to save")
    torch.save(generator_adapter.corrector.state_dict_trainable(), ckpt_dir / "corrector.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    if trainer_state is not None:
        (ckpt_dir / "trainer_state.json").write_text(
            json.dumps(trainer_state, indent=2) + "\n", encoding="utf-8"
        )
    (ckpt_dir / "manifest.json").write_text(
        json.dumps({"protocol": "reflective_t2i_v1", "step": int(step), "config": config_payload}, indent=2) + "\n",
        encoding="utf-8",
    )
    if int(keep_checkpoints) > 0:
        existing = sorted(
            [d for d in out_path.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda p: p.name,
        )
        for old in existing[: max(0, len(existing) - int(keep_checkpoints))]:
            shutil.rmtree(old, ignore_errors=True)
    return ckpt_dir


def load_corrector_checkpoint(
    ckpt_dir: str | Path,
    generator_adapter,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, dict[str, Any]]:
    cpath = Path(ckpt_dir)
    sd = torch.load(cpath / "corrector.pt", map_location="cpu", weights_only=True)
    generator_adapter.corrector.load_state_dict_trainable(sd, strict=True)
    if optimizer is not None and (cpath / "optimizer.pt").exists():
        opt_sd = torch.load(cpath / "optimizer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(opt_sd)
    manifest = json.loads((cpath / "manifest.json").read_text(encoding="utf-8"))
    tstate_path = cpath / "trainer_state.json"
    tstate = json.loads(tstate_path.read_text(encoding="utf-8")) if tstate_path.exists() else {}
    return int(manifest["step"]), tstate


def train_corrector(
    generator_adapter,
    dataset,
    train_cfg: TrainConfig,
    *,
    output_dir: str | Path,
    max_steps: int,
    rank: int = 0,
    world_size: int = 1,
    resume_from: str | Path | None = None,
    config_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Production single- or multi-node DDP training loop for any GeneratorAdapter."""
    if generator_adapter.corrector is None:
        raise RuntimeError("Attach a StateCorrector to generator_adapter before training")
    out = Path(output_dir).resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    params = list(generator_adapter.corrector.trainable_parameters().values())
    optimizer = torch.optim.AdamW(
        params,
        lr=float(train_cfg.lr),
        betas=tuple(float(b) for b in train_cfg.betas),
        weight_decay=float(train_cfg.weight_decay),
        foreach=False,
    )

    start_step = 0
    cursor = 0
    epoch = 0
    if resume_from:
        start_step, tstate = load_corrector_checkpoint(resume_from, generator_adapter, optimizer=optimizer)
        cursor = int(tstate.get("cursor", 0))
        epoch = int(tstate.get("epoch", 0))
    _broadcast_trainable(params)

    loss_fn = ReflectiveCorrectionLoss(train_cfg.loss)
    null_cond = generator_adapter.encode_prompt(None)
    py_rng = random.Random(int(train_cfg.seed) + 100003 * (rank + 1) + start_step * 997)
    t_gen = torch.Generator(device=generator_adapter.device).manual_seed(int(train_cfg.seed) + rank + start_step)

    n_rows = len(dataset)
    order = list(range(rank, n_rows, max(1, world_size)))
    epoch_rng = random.Random(int(train_cfg.seed) * 7 + epoch + rank * 13)
    epoch_rng.shuffle(order)

    grad_accum = max(1, int(train_cfg.grad_accum))
    global_batch = max(1, int(world_size)) * grad_accum
    log_path = out / "train.jsonl"
    last_loss = None

    for step in range(start_step + 1, int(max_steps) + 1):
        t0 = time.perf_counter()
        lr = cosine_lr(step, int(max_steps), train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_wclean = 0.0
        accum_cos = 0.0
        accum_soar = 0
        accum_drop = 0
        last_metrics: dict[str, Any] = {}

        for _ in range(grad_accum):
            if cursor >= len(order):
                epoch += 1
                epoch_rng = random.Random(int(train_cfg.seed) * 7 + epoch + rank * 13)
                epoch_rng.shuffle(order)
                cursor = 0
            sample = dataset[order[cursor]]
            cursor += 1

            loss, metrics = training_step(
                generator_adapter,
                sample,
                loss_fn,
                train_cfg,
                step=step,
                null_cond=null_cond,
                rng=py_rng,
                torch_generator=t_gen,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at rank={rank} step={step}: {metrics}")
            (loss / float(grad_accum)).backward()
            accum_loss += float(loss.detach().item()) / float(grad_accum)
            accum_wclean += float(metrics.get("loss_wclean", metrics.get("loss_clean", 0.0))) / float(grad_accum)
            accum_cos += float(metrics.get("loss_cos", 0.0)) / float(grad_accum)
            accum_soar += int(metrics.get("soar_used", 0))
            accum_drop += int(metrics.get("prompt_dropped", 0))
            last_metrics = metrics

        _sync_gradients(params)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(train_cfg.gradient_clip)))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite grad_norm at rank={rank} step={step}")
        optimizer.step()

        # Average scalar loss across ranks for accurate rank-0 logging
        loss_tensor = torch.tensor([accum_loss], device=generator_adapter.device, dtype=torch.float32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor.div_(float(dist.get_world_size()))
        mean_loss = float(loss_tensor.item())
        last_loss = mean_loss
        sec = time.perf_counter() - t0

        if rank == 0:
            row = {
                "step": step,
                "epoch": epoch,
                "lr": lr,
                "loss": round(mean_loss, 6),
                "loss_wclean": round(accum_wclean, 6),
                "loss_cos": round(accum_cos, 6),
                "delta_cos": round(float(last_metrics.get("delta_cos", 0.0)), 4),
                "edit_weight_ratio": round(float(last_metrics.get("edit_weight_ratio", 1.0)), 3),
                "soar_frac": round(accum_soar / float(grad_accum), 3),
                "drop_frac": round(accum_drop / float(grad_accum), 3),
                "grad_norm": round(grad_norm, 5),
                "global_batch": global_batch,
                "seconds": round(sec, 4),
                "samples_per_sec": round(global_batch / max(sec, 1e-4), 2),
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if step == 1 or step % max(1, int(train_cfg.log_every)) == 0 or step == int(max_steps):
                print(
                    f"[step {step:05d}/{int(max_steps):05d}] loss={mean_loss:.5f} "
                    f"(wclean={accum_wclean:.5f} cos={accum_cos:.4f}) "
                    f"gnorm={grad_norm:.3f} lr={lr:.2e} {row['samples_per_sec']:.1f} smp/s",
                    flush=True,
                )
            if step % int(train_cfg.save_every) == 0 or step == int(max_steps):
                save_corrector_checkpoint(
                    out,
                    generator_adapter,
                    step,
                    config_payload or {},
                    optimizer=optimizer,
                    trainer_state={"cursor": cursor, "epoch": epoch},
                    keep_checkpoints=train_cfg.keep_checkpoints,
                )

    if dist.is_initialized():
        dist.barrier()
    summary = {
        "protocol": "reflective_t2i_v1",
        "steps": int(max_steps),
        "world_size": int(world_size),
        "grad_accum": grad_accum,
        "global_batch": global_batch,
        "final_loss": last_loss,
        "dataset_size": n_rows,
    }
    if rank == 0:
        (out / "complete.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
