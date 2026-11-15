"""T2I with UND feedback, one prompt per rank.

Each case walks the deployment path for real:

1. native T2I from noise to the read time;
2. a lookahead preview, which is the only picture the diagnoser sees;
3. the three-step reflection, which either says "already satisfied" or writes
   one executable instruction;
4. when there is an instruction, the editor moves the state at the fixed time
   and the frozen tail finishes the picture.

Three pictures are saved per case so a reader can see what the feedback did:
the preview, the edited state's lookahead, and the finished picture. The
reflection itself is stored verbatim.

    torchrun --nnodes=1 --nproc_per_node=16 t2i_feedback_bench.py ...
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.diagnosis import review_reflection
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, TailT2I, edit_state, euler, phase_grid, preview_image, segment_times,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor, TailGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--edit-cfg-scale', type=float, default=None)
    parser.add_argument('--tail-checkpoint')
    parser.add_argument('--tail-config')
    args = parser.parse_args()

    from qwen_latent_cot.u15_target_correction import distributed as dist
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    device = set_device(args.device) if world < 2 else dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, world > 1)
    grid = phase_grid(rt, cfg)
    read = float(cfg['read_time'])
    steps = int(cfg['edit_steps'])
    edit_cfg_scale = float(cfg.get('edit_cfg_scale', 1.0) if args.edit_cfg_scale is None else args.edit_cfg_scale)
    sigma = rt.noise_scale()
    editor = StateEditor(rt, cfg).eval()
    load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg, component='editor'))
    tail = None
    if args.tail_checkpoint:
        tail_cfg = load_config(args.tail_config) if args.tail_config else dict(cfg, component='tail')
        tail = TailGenerator(rt, tail_cfg).eval()
        load_stage_weights(args.tail_checkpoint, tail, editor_identity(rt, tail_cfg, component='tail'))

    pool = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    cases = pool[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    shard = args.out / f'shard_{rank:02d}.jsonl'
    images_dir = args.out / f'img_{rank:02d}'
    images_dir.mkdir(exist_ok=True)
    tail_times = segment_times(grid, read, 1.0)

    mine = [(position, case) for position, case in enumerate(cases) if position % world == rank]
    for position, item in mine:
        began = time.perf_counter()
        raw = item['prompt']
        index = position
        seed = int(args.seed) + position
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                            generator=generator) * sigma

        native = NativeT2I(rt, raw, cfg)
        state = euler(noise.float(), segment_times(grid, 0.0, read), native)
        preview = preview_image(state, read, grid, NativeT2I(rt, raw, cfg))
        tag = f'{index:03d}'
        save_image(images_dir / f'{tag}_preview.png', preview)

        # The same reflection the standalone baseline measured, now inside the
        # loop: describe, judge, and only then instruct.
        review = review_reflection(rt, preview, raw, cfg)['reflection']

        edited = None
        edited_preview = None
        if review['outcome'] == 'edit':
            prefixes = editor.prefixes(review['instruction'])
            base = editor.image_clean(state, read, prefixes['image'])
            # edit_state takes prefixes['joint'] as conditional and
            # prefixes['empty'] as unconditional; overriding joint with the
            # empty prefix would zero the CFG term and drop the instruction.
            edited, _ = edit_state(
                editor, state, read, prefixes, steps,
                base_prediction=base, cfg_scale=edit_cfg_scale,
            )
            raw_native = NativeT2I(rt, raw, cfg)
            tail_field = raw_native if tail is None else TailT2I(tail, raw_native, cfg)
            edited_preview = preview_image(edited, read, grid, tail_field)
            save_image(images_dir / f'{tag}_edited_preview.png', edited_preview)
            result = euler(edited, tail_times, tail_field)
        else:
            result = euler(state, tail_times, NativeT2I(rt, raw, cfg))
        save_image(images_dir / f'{tag}_result.png', result)

        row = dict(index=index, prompt_id=item.get('prompt_id'), prompt=raw,
                   skills=item.get('skills'), vqa_list=item.get('vqa_list'), seed=seed,
                   read_time=read, edit_steps=steps, checkpoint=str(args.checkpoint),
                   description=review['description'], match=review['match'],
                   instruction=review['instruction'], preserve=review['preserve'],
                   outcome=review['outcome'], raw_response=review['raw'][:4000],
                   seconds=time.perf_counter() - began)
        with shard.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(dict(rank=rank, index=index, outcome=review['outcome'],
                              match=review['match']), ensure_ascii=False), flush=True)
        del preview, edited_preview, result, state, edited

    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    rows = []
    for path in sorted(args.out.glob('shard_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    rows.sort(key=lambda r: r['index'])
    (args.out / 'cases.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    counts = {}
    for row in rows:
        counts[row['outcome']] = counts.get(row['outcome'], 0) + 1
    print(json.dumps(dict(cases=len(rows), outcomes=counts,
                          mean_seconds=sum(r['seconds'] for r in rows) / max(len(rows), 1)),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
