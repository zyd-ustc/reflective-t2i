"""Collect real editor rollout terminal states q_K for standalone tail training."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction import distributed as dist
from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.data import classify_semantic_task, read_rows
from qwen_latent_cot.u15_target_correction.field import NativeT2I, edit_state, euler, phase_grid, segment_times
from qwen_latent_cot.u15_target_correction.images import load_pair
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor


def _load_captions(paths):
    captions = {}
    for path in paths or ():
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = str(row.get('uid') or '').strip()
            text = (row.get('caption') or row.get('target_caption') or row.get('tail_condition') or '').strip()
            if uid and text:
                captions[uid] = text
    return captions


def _resolve_existing(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _collect_records(manifest_paths, split):
    rows = []
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).resolve()
        root = manifest_path.parent
        for row in read_rows(manifest_path):
            row_split = row.get('split', 'train')
            if split != 'all' and row_split != split:
                continue
            copy = dict(row)
            copy['_source_root'] = str(root)
            copy['source_image'] = str(_resolve_existing(root, row['source_image']))
            copy['target_image'] = str(_resolve_existing(root, row['target_image']))
            rows.append(copy)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', nargs='+', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--split', default='train', choices=('train', 'val', 'test', 'all'))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--edit-cfg-scale', type=float, default=None)
    parser.add_argument('--start-mode', choices=('linear', 't2i'), default='linear')
    parser.add_argument('--captions', nargs='*', default=())
    parser.add_argument('--target-captions', nargs='*', default=(),
                        help='Optional JSONL(s) providing uid -> target caption for tail_condition')
    parser.add_argument('--device', default='npu:0')
    args = parser.parse_args()

    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = set_device(args.device) if world < 2 else dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    read = float(cfg['read_time'])
    steps = int(cfg['edit_steps'])
    edit_cfg_scale = float(cfg.get('edit_cfg_scale', 1.0) if args.edit_cfg_scale is None else args.edit_cfg_scale)

    out = args.out.resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'states').mkdir(exist_ok=True)
        (out / 'shards').mkdir(exist_ok=True)
    if world > 1:
        dist.barrier()

    rt = load_native(args.model, args.source, cfg, device, world > 1)
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg, component='editor'))
    grid = phase_grid(rt, cfg)
    sigma = rt.noise_scale()

    source_captions = _load_captions(args.captions)
    target_captions = _load_captions(args.target_captions)
    records = _collect_records(args.manifest, args.split)
    if args.limit and args.limit < len(records):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(records)), args.limit))
        records = [records[i] for i in indices]

    shard_path = out / 'shards' / f'rank_{rank:04d}.jsonl'
    mine = [(idx, rec) for idx, rec in enumerate(records) if idx % world == rank]
    with shard_path.open('w') as shard_file:
        for idx, rec in mine:
            uid = str(rec['uid'])
            bucket = hashlib.sha256(uid.encode()).hexdigest()[:2]
            rel_state = Path('states') / bucket / f'{idx:07d}_{uid}.pt'
            state_path = out / rel_state
            state_path.parent.mkdir(parents=True, exist_ok=True)

            if not state_path.exists():
                source, _, _, _ = load_pair(rec, rt.width, rt.height, rt.device)
                gen = torch.Generator(device=rt.device).manual_seed(int(args.seed) + idx)
                noise = torch.randn(source.shape, device=rt.device, generator=gen, dtype=torch.float32) * sigma
                if args.start_mode == 't2i':
                    scene = (rec.get('parent_condition') or source_captions.get(uid) or '').strip()
                    if not scene:
                        raise ValueError('t2i start mode requires parent_condition or --captions for uid=' + uid)
                    native = NativeT2I(rt, scene, cfg)
                    anchor = euler(noise, segment_times(grid, 0.0, read), native)
                else:
                    anchor = read * source + (1.0 - read) * noise
                prefixes = editor.prefixes(rec['edit_instruction'])
                base = editor.image_clean(anchor, read, prefixes['image'])
                # Pass no joint override: edit_state takes prefixes['joint'] as the
                # conditional branch and prefixes['empty'] as the unconditional one.
                # Overriding joint with the empty prefix zeroed the CFG term, so a
                # collected q_K carried no trace of the instruction.
                q_k, _ = edit_state(
                    editor,
                    anchor,
                    read,
                    prefixes,
                    steps,
                    base_prediction=base,
                    cfg_scale=edit_cfg_scale,
                )
                torch.save(q_k.squeeze(0).to(dtype=torch.bfloat16).cpu(), state_path)

            semantic_task, count_direction = classify_semantic_task(rec.get('edit_instruction'), rec.get('task'))
            tail_cond = (rec.get('tail_condition') or target_captions.get(uid) or '').strip()
            out_row = dict(
                uid=uid,
                split=rec.get('split', 'train'),
                source_image=rec['source_image'],
                target_image=rec['target_image'],
                tail_state_path=str(rel_state),
                edit_instruction=rec['edit_instruction'],
                tail_condition=tail_cond,
                geometry=rec.get('geometry'),
                task=rec.get('task'),
                semantic_task=rec.get('semantic_task') or semantic_task,
                count_direction=rec.get('count_direction') or count_direction,
                is_pure_t2i=False,
            )
            shard_file.write(json.dumps(out_row, ensure_ascii=False) + '\n')

    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = []
        for r in range(world):
            spath = out / 'shards' / f'rank_{r:04d}.jsonl'
            if spath.exists():
                merged.extend(json.loads(line) for line in spath.read_text().splitlines() if line.strip())
        manifest = dict(
            kind='tail_states',
            read_time=read,
            edit_steps=steps,
            edit_cfg_scale=edit_cfg_scale,
            start_mode=args.start_mode,
            checkpoint=str(args.checkpoint),
            records=merged,
        )
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    if world > 1:
        dist.shutdown()


if __name__ == '__main__':
    main()
