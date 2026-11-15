"""Configuration for Stage 1 and v2 rollout / GEN-full / tail training recipes."""

import json
import math
from pathlib import Path

from . import CONDITION_PROTOCOL, KIND, PROTOCOL_VERSION

VALID_PARAMETERIZATIONS = ("gen_lora", "gen_full")
VALID_COMPONENTS = ("editor", "tail")


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    parent = cfg.pop('extends', None)
    if parent:
        base = json.loads((path.parent / parent).read_text())
        if 'extends' in base:
            raise ValueError('A shared core cannot extend another configuration')
        base.update(cfg)
        cfg = base
    validate_config(cfg)
    return cfg


def _validate_prob(cfg, key, *, strict_less_than_one=False):
    value = float(cfg.get(key, 0.0))
    if value < 0.0 or (value >= 1.0 if strict_less_than_one else value > 1.0):
        raise ValueError(f"{key} must be a valid probability")
    return value


def _validate_stage1_strict(cfg):
    if cfg.get('source_weights') != {'linear_pair': 1.0}:
        raise ValueError('Only Stage 1 linear-pair training is supported')
    if cfg.get('role_weights') != {'L_edit': 1.0} or cfg.get('keep_fraction') != 0:
        raise ValueError('The Stage 1 recipe requires only L_edit')
    for key in ('terminal', 'semantic_weights', 'recovery_prob'):
        if cfg.get(key):
            raise ValueError(key + ' is not part of the Stage 1 recipe')


def validate_config(cfg):
    if cfg.get('kind') != KIND or cfg.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Isochronic state flow configuration required')
    stage = cfg.get('stage')
    if stage not in (1, 2):
        raise ValueError('Only Stage 1 and Stage 2 configurations are supported')
    component = str(cfg.get('component', 'editor'))
    if component not in VALID_COMPONENTS:
        raise ValueError('component must be editor or tail')
    if stage == 1 and component != 'editor':
        raise ValueError('Stage 1 only supports the editor component')
    if stage == 1:
        _validate_stage1_strict(cfg)
    if cfg.get('condition_protocol') != CONDITION_PROTOCOL:
        raise ValueError('Condition protocol must match the Stage 1 checkpoint')
    for key in ('width', 'height', 'num_steps', 'edit_steps', 'lora_rank',
                'global_batch', 'save_every'):
        if not isinstance(cfg.get(key), int) or cfg[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if cfg['width'] % 8 or cfg['height'] % 8:
        raise ValueError('Canvas dimensions must be divisible by eight')
    if cfg['cfg_norm'] != 'none' or float(cfg['cfg_scale']) != 4 or float(cfg['image_cfg_scale']) != 1:
        raise ValueError('Native guidance must match the Stage 1 recipe')
    lo, hi = map(float, cfg['cfg_interval'])
    if not 0 <= lo < hi <= 1 or not 0 < float(cfg['t_eps']) < 1:
        raise ValueError('Invalid native sampling interval')
    times = list(map(float, cfg['state_times']))
    probs = list(map(float, cfg['state_time_probs']))
    if times != [float(cfg['read_time'])] or not 0 < times[0] < 1 or probs != [1.0]:
        raise ValueError('Editor and tail recipes use exactly one read time')
    if int(cfg['lookahead_steps']) != 2 or float(cfg['timestep_shift']) <= 0:
        raise ValueError('Invalid preview or timestep schedule')
    if float(cfg['lr']) <= 0 or not 0 < float(cfg['lr_final']) <= float(cfg['lr']):
        raise ValueError('Invalid learning-rate schedule')
    if int(cfg['warmup_steps']) < 0 or float(cfg['gradient_clip']) <= 0:
        raise ValueError('Invalid training schedule')

    param_mode = str(cfg.get('parameterization', 'gen_lora'))
    if param_mode not in VALID_PARAMETERIZATIONS:
        raise ValueError('parameterization must be gen_lora or gen_full')
    tail_param_mode = str(cfg.get('tail_parameterization', 'gen_lora'))
    if tail_param_mode not in VALID_PARAMETERIZATIONS:
        raise ValueError('tail_parameterization must be gen_lora or gen_full')
    if int(cfg.get('tail_lora_rank', cfg['lora_rank'])) < 1:
        raise ValueError('tail_lora_rank must be positive')

    rollout_prob = _validate_prob(cfg, 'rollout_prob')
    terminal_rollout_prob = _validate_prob(cfg, 'terminal_rollout_prob')
    if rollout_prob + terminal_rollout_prob > 1.0 + 1e-9:
        raise ValueError('rollout_prob + terminal_rollout_prob cannot exceed 1.0')
    _validate_prob(cfg, 'edit_prompt_dropout', strict_less_than_one=True)
    _validate_prob(cfg, 'rollout_cfg_prob')
    _validate_prob(cfg, 'keep_fraction', strict_less_than_one=True)
    _validate_prob(cfg, 'tail_native_prob', strict_less_than_one=True)
    _validate_prob(cfg, 'tail_prompt_dropout', strict_less_than_one=True)

    if float(cfg.get('edit_cfg_scale', 1.0)) < 1.0 or float(cfg.get('rollout_cfg_scale', 1.0)) < 1.0:
        raise ValueError('Edit CFG scale must be at least 1.0')
    t_min = float(cfg.get('tail_t_min', cfg['read_time']))
    t_max = float(cfg.get('tail_t_max', 1.0))
    if not 0.0 < t_min < t_max <= 1.0:
        raise ValueError('Invalid tail time interval')
    if cfg.get('task_weights'):
        # Traversal is shuffled and record-equal; a per-task ratio would reintroduce
        # the hand-tuned mixture this recipe deliberately dropped.
        raise ValueError(
            'task_weights is retired: every record is traversed once per epoch at equal weight'
        )


def stage_updates(records, cfg, global_batch=None):
    """Steps needed to traverse the training set ``epochs`` times.

    One epoch is ceil(records / global_batch) updates, because every rank draws a
    disjoint record per step. The old ``max_steps`` cap is gone on purpose: it was
    sized for a 490k set and would silently truncate a 15M traversal.
    """
    records = int(records)
    batch = int(global_batch or cfg['global_batch'])
    if records < 1 or batch < 1:
        raise ValueError('Training needs records and a positive global batch')
    epochs = int(cfg.get('epochs', 1))
    if epochs < 1:
        raise ValueError('epochs must be a positive integer')
    return epochs * math.ceil(records / batch)
