"""Configuration for the strict isochronic manifold-transition LatentCoT operator."""

import json
import math
from pathlib import Path

from . import CONDITION_PROTOCOL, EDITOR_INTERFACE, KIND, PROTOCOL_VERSION

RETIRED_KEYS = (
    "edit_steps",
    "rollout_prob",
    "terminal_rollout_prob",
    "lambda_distill",
    "time_curriculum",
    "tail_parameterization",
    "tail_lora_rank",
    "tail_t_min",
    "tail_t_max",
    "tail_native_prob",
    "tail_prompt_dropout",
    "edit_loss_type",
    "unboxed_policy",
    "real_state_mix_prob",
    "task_weights",
)


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        base = json.loads((path.parent / parent).read_text())
        if "extends" in base:
            raise ValueError("A shared core cannot extend another configuration")
        base.update(cfg)
        cfg = base
    validate_config(cfg)
    return cfg


def _validate_prob(cfg, key, *, strict_less_than_one=False, default=0.0):
    value = float(cfg.get(key, default))
    if value < 0.0 or (value >= 1.0 if strict_less_than_one else value > 1.0):
        raise ValueError(f"{key} must be a valid probability")
    return value


def validate_config(cfg):
    if cfg.get("kind") != KIND or cfg.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Isochronic manifold configuration required")
    for retired in RETIRED_KEYS:
        if retired in cfg and cfg[retired] not in (None, {}, []):
            raise ValueError(f"{retired} belongs to a retired architecture and is no longer supported")
    if cfg.get("component", "editor") != "editor":
        raise ValueError("Only the unified isochronic manifold editor is supported (tail generator is retired)")
    if int(cfg.get("stage", 1)) != 1:
        raise ValueError("Only stage=1 unified isochronic manifold training is supported")
    if str(cfg.get("editor_interface")) != EDITOR_INTERFACE:
        raise ValueError(f"editor_interface must be {EDITOR_INTERFACE!r}")
    if str(cfg.get("condition_protocol")) != CONDITION_PROTOCOL:
        raise ValueError(f"condition_protocol must be {CONDITION_PROTOCOL!r}")
    if str(cfg.get("parameterization", "gen_lora")) != "gen_lora":
        raise ValueError("Only parameterization='gen_lora' is supported")

    for key in ("width", "height", "num_steps", "lora_rank", "global_batch", "save_every"):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["width"] % 8 or cfg["height"] % 8:
        raise ValueError("Canvas dimensions must be divisible by eight")

    if cfg["cfg_norm"] != "none" or float(cfg["cfg_scale"]) != 4.0 or float(cfg["image_cfg_scale"]) != 1.0:
        raise ValueError("Native T2I guidance must keep cfg_scale=4.0, image_cfg_scale=1.0, cfg_norm='none'")
    lo, hi = map(float, cfg["cfg_interval"])
    if not 0.0 <= lo < hi <= 1.0 or not 0.0 < float(cfg["t_eps"]) < 1.0:
        raise ValueError("Invalid native sampling interval")

    t0 = float(cfg["read_time"])
    t_stop = float(cfg["stop_time"])
    t_train_max = float(cfg.get("train_max_time", t_stop))
    if not 0.0 < t0 < t_stop < 1.0:
        raise ValueError("Require 0 < read_time < stop_time < 1.0 for strict isochronic manifold transition")
    if not t0 < t_train_max <= 1.0:
        raise ValueError("Require read_time < train_max_time <= 1.0")
    if abs(float(cfg.get("bridge_endpoint_time", 1.0)) - 1.0) > 1e-9:
        raise ValueError("bridge_endpoint_time must be 1.0 (clean target endpoint)")

    _validate_prob(cfg, "bridge_start_prob", default=0.15)
    _validate_prob(cfg, "noise_refresh", default=0.0)
    _validate_prob(cfg, "edit_prompt_dropout", strict_less_than_one=True, default=0.0)
    if float(cfg.get("edit_cfg_scale", 1.0)) < 1.0:
        raise ValueError("edit_cfg_scale must be >= 1.0")
    if float(cfg.get("keep_fraction", 0.0)) != 0.0:
        raise ValueError("keep_fraction != 0 is retired")

    lookahead = int(cfg["lookahead_steps"])
    if lookahead < 2 or float(cfg["timestep_shift"]) <= 0.0:
        raise ValueError("lookahead_steps must be an integer >= 2 and timestep_shift > 0")
    for flag in ("recheck_add_on_preview", "gate_contradictory_edits"):
        if not isinstance(cfg.get(flag, True), bool):
            raise ValueError(f"{flag} must be a boolean")

    if float(cfg["lr"]) <= 0.0 or not 0.0 < float(cfg["lr_final"]) <= float(cfg["lr"]):
        raise ValueError("Invalid learning-rate schedule")
    if int(cfg["warmup_steps"]) < 0 or float(cfg["gradient_clip"]) <= 0.0:
        raise ValueError("Invalid warmup_steps or gradient_clip")


def stage_updates(records, cfg, global_batch=None):
    """Steps needed to traverse the training set ``epochs`` times."""
    records = int(records)
    batch = int(global_batch or cfg["global_batch"])
    if records < 1 or batch < 1:
        raise ValueError("Training needs records and a positive global batch")
    epochs = int(cfg.get("epochs", 1))
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    return epochs * math.ceil(records / batch)
