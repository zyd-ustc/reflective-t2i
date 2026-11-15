"""The album test again, on held-out data: does the editor generalise or memorise?

The album's 32 cases all came from ``split: train``, so its successes cannot be
read as capability. This runs the same construction on ``split: val`` records —
same source/target pair, same instruction wording, same eight-step edit — where
nothing was ever optimised.

Two starts per case:

``linear``  t * x_src + (1 - t) * sigma * eps, the way training built it;
``t2i``     the state the model's own T2I trajectory holds at the read time.

The finished picture is saved for both. A held-out case that edits correctly
means the operator generalises; a train-only success would mean it memorises.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import torch.nn.functional as F

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, TailT2I, edit_state, euler, phase_grid, preview_image, segment_times,
)
from qwen_latent_cot.u15_target_correction.images import load_canvas, save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor, TailGenerator

# The tail is a text-conditioned model and every L record lacks a
# target caption, so the continuation falls back to None (unconditional, matching inference.py).
SCENE_FALLBACK = None


def _lowfreq(tensor, pool=16):
    return F.avg_pool2d(tensor.detach().float(), kernel_size=pool, stride=pool).reshape(-1)


def _displacement_metrics(disp, target_disp):
    """Direction, magnitude ratio, and projected progress along target_disp."""
    a_full, b_full = disp.detach().float().reshape(-1), target_disp.detach().float().reshape(-1)
    a_low, b_low = _lowfreq(disp), _lowfreq(target_disp)
    cos_full = float(F.cosine_similarity(a_full, b_full, dim=0))
    cos_low = float(F.cosine_similarity(a_low, b_low, dim=0))
    b_full_norm = float(b_full.norm()) + 1e-8
    b_low_norm = float(b_low.norm()) + 1e-8
    disp_ratio_full = float(a_full.norm()) / b_full_norm
    disp_ratio_low = float(a_low.norm()) / b_low_norm
    proj_ratio_full = float(torch.dot(a_full, b_full)) / (b_full_norm ** 2)
    proj_ratio_low = float(torch.dot(a_low, b_low)) / (b_low_norm ** 2)
    return dict(
        cos_full=cos_full,
        cos_low=cos_low,
        disp_ratio_full=disp_ratio_full,
        disp_ratio_low=disp_ratio_low,
        proj_ratio_full=proj_ratio_full,
        proj_ratio_low=proj_ratio_low,
    )


def _family(instruction):
    """Coarse edit family, so the sample is not one hard corner of the data."""
    text = (instruction or '').lower()
    if text.startswith(('add ', 'remove ')):
        return 'count'
    if text.startswith(('rearrange', 'swap', 'move')):
        return 'spatial'
    if text.startswith('replace'):
        return 'replace'
    if text.startswith('change'):
        return 'attribute'
    return 'other'


def _pick(records, limit, seed, families=None):
    """Held-out records spread across edit families, not just counting.

    An earlier version kept only counting edits, which are the hardest family
    at t=0.36; the failure it measured was partly the choice of task.
    """
    import random
    buckets = {}
    for record in records:
        family = _family(record.get('edit_instruction'))
        if families and family not in families:
            continue
        buckets.setdefault(family, []).append(record)
    rng = random.Random(seed)
    for values in buckets.values():
        values.sort(key=lambda r: r['uid'])
        rng.shuffle(values)
    chosen, index = [], 0
    # Round-robin across families so no single edit type dominates.
    while len(chosen) < limit and any(buckets.values()):
        for family in sorted(buckets):
            if buckets[family] and len(chosen) < limit:
                chosen.append(buckets[family].pop())
        index += 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--split', default='val')
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--captions', type=Path,
                        help='JSONL of uid->caption for the T2I start scene text')
    parser.add_argument('--edit-cfg-scale', type=float, default=None)
    parser.add_argument('--tail-checkpoint')
    parser.add_argument('--tail-config')
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()

    from qwen_latent_cot.u15_target_correction import distributed as dist
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = set_device(args.device) if world < 2 else dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, world > 1)
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg, component='editor'))
    edit_cfg_scale = float(cfg.get('edit_cfg_scale', 1.0) if args.edit_cfg_scale is None else args.edit_cfg_scale)
    tail = None
    if args.tail_checkpoint:
        tail_cfg = load_config(args.tail_config) if args.tail_config else dict(cfg, component='tail')
        tail = TailGenerator(rt, tail_cfg).eval()
        load_stage_weights(args.tail_checkpoint, tail, editor_identity(rt, tail_cfg, component='tail'))
    grid = phase_grid(rt, cfg)
    read = float(cfg['read_time'])
    steps = int(cfg['edit_steps'])
    sigma = rt.noise_scale()
    tail_times = segment_times(grid, read, 1.0)

    # Scene text for the T2I start: the cache has no parent_condition, so the
    # captions are generated from the source images ahead of time.
    captions = {}
    if args.captions:
        for line in args.captions.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('caption'):
                    captions[row['uid']] = row['caption']
    records = [r for r in json.loads(args.manifest.read_text())['records']
               if r['split'] == args.split]
    cases = _pick(records, args.limit, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    shard = args.out / f'shard_{rank:02d}.jsonl'
    images_dir = args.out / f'img_{rank:02d}'
    images_dir.mkdir(exist_ok=True)

    mine = [(i, c) for i, c in enumerate(cases) if i % world == rank]
    for position, record in mine:
        began = time.perf_counter()
        index = position
        source, _, _ = load_canvas(Path(record['source_image']), rt.width, rt.height, rt.device)
        target, _, _ = load_canvas(Path(record['target_image']), rt.width, rt.height, rt.device)
        save_image(images_dir / f'{index:02d}_src.png', source)
        save_image(images_dir / f'{index:02d}_gt.png', target)

        generator = torch.Generator(device=rt.device).manual_seed(args.seed + index)
        unit = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator)
        noise = unit * sigma
        linear_state = read * source + (1 - read) * noise
        target_state = read * target + (1 - read) * noise
        true_disp = target_state - linear_state

        # The deployment start: a plain T2I trajectory. With no scene caption in
        # the L cache, None (unconditional) avoids passing empty-text CFG=4.
        # The deployment start needs a scene description, which the L cache
        # lacks. --captions supplies one generated from the source image; the
        # record's own parent_condition is used when a cache has it.
        scene = (record.get('parent_condition')
                 or (captions.get(record['uid']) if captions else None))
        if not scene:
            raise ValueError('A T2I start needs a scene description for ' + record['uid'])
        native = NativeT2I(rt, scene, cfg)
        t2i_state = euler(noise.float(), segment_times(grid, 0.0, read), native)

        instruction = record['edit_instruction']
        family = _family(instruction)
        cells = {}
        linear_metrics = {}
        for tag, state in (('linear', linear_state), ('t2i', t2i_state)):
            prefixes = editor.prefixes(instruction)
            base = editor.image_clean(state, read, prefixes['image'])
            if tag == 'linear':
                s0_vel = editor.state_velocity(state, read, 0.0, prefixes['joint'], base)
                linear_metrics['s0'] = _displacement_metrics(s0_vel, true_disp)
            # edit_state takes prefixes['joint'] as conditional and
            # prefixes['empty'] as unconditional; overriding joint with the
            # empty prefix would zero the CFG term and drop the instruction.
            edited, _ = edit_state(
                editor, state, read, prefixes, steps,
                base_prediction=base, cfg_scale=edit_cfg_scale,
            )
            if tag == 'linear':
                linear_metrics['end'] = _displacement_metrics(edited - state, true_disp)
            # The tail is a text-conditioned generator: handing it the instruction
            # makes it paint "so that it contains exactly 3" as picture content.
            # The L cache carries no target caption, so fall back to None (unconditional).
            tail_prompt = record.get('tail_condition') or scene
            tail_native = NativeT2I(rt, tail_prompt, cfg)
            tail_field = tail_native if tail is None else TailT2I(tail, tail_native, cfg)
            lookahead = preview_image(edited, read, grid, tail_field)
            result = euler(edited, tail_times, tail_field)
            save_image(images_dir / f'{index:02d}_{tag}_preview.png', lookahead)
            save_image(images_dir / f'{index:02d}_{tag}_result.png', result)
            cells[tag] = float((edited - state).abs().mean())

        row = dict(index=index, uid=record['uid'], split=record['split'],
                   family=family, instruction=instruction, task=record.get('task'),
                   displacement=cells, linear_metrics=linear_metrics,
                   seconds=time.perf_counter() - began)
        with shard.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(dict(index=index, uid=record['uid'], family=family,
                              s0_cos_low=round(linear_metrics['s0']['cos_low'], 3),
                              end_cos_low=round(linear_metrics['end']['cos_low'], 3),
                              disp_ratio_low=round(linear_metrics['end']['disp_ratio_low'], 3),
                              proj_ratio_low=round(linear_metrics['end']['proj_ratio_low'], 3),
                              disp_linear=round(cells['linear'], 5),
                              disp_t2i=round(cells['t2i'], 5)), ensure_ascii=False), flush=True)

    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    rows = []
    for path in sorted(args.out.glob('shard_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: r['index'])
    (args.out / 'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    def _agg(items):
        n = len(items)
        if not n:
            return {}
        return dict(
            cases=n,
            mean_linear=sum(r['displacement']['linear'] for r in items) / n,
            mean_t2i=sum(r['displacement']['t2i'] for r in items) / n,
            s0_cos_low=sum(r['linear_metrics']['s0']['cos_low'] for r in items) / n,
            end_cos_low=sum(r['linear_metrics']['end']['cos_low'] for r in items) / n,
            disp_ratio_low=sum(r['linear_metrics']['end']['disp_ratio_low'] for r in items) / n,
            proj_ratio_low=sum(r['linear_metrics']['end']['proj_ratio_low'] for r in items) / n,
        )

    summary = dict(overall=_agg(rows), by_family={})
    for fam in sorted({r['family'] for r in rows}):
        summary['by_family'][fam] = _agg([r for r in rows if r['family'] == fam])
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
