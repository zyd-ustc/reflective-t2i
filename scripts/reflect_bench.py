"""Run the three-step reflection over N prompts, one case per rank.

Each rank walks its own prompt to the read time, renders the preview the
diagnoser actually sees, and records what the frozen UND answers:

* ``description`` — the model's own words about the picture
* ``match``       — its yes/no verdict on whether the prompt is satisfied
* ``instruction`` — the executable edit it proposes when the answer is no

Nothing here scores or parses those answers: the page built from the output
shows the preview next to the three fields so a reader can judge them. The
only text handling is ``diagnosis.parse_reflection``, which splits the labelled
lines apart; it never interprets their content.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.diagnosis import review_reflection
from qwen_latent_cot.u15_target_correction.field import (
    NativeT2I, euler, phase_grid, preview_image, segment_times, time_key,
)
from qwen_latent_cot.u15_target_correction.images import save_image
from qwen_latent_cot.u15_target_correction.loading import load_native

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    from qwen_latent_cot.u15_target_correction import distributed as dist
    # Read the launcher's own world size; torchrun sets RANK/WORLD_SIZE per
    # process, and a stale value in the environment would send every rank to
    # the same chip.
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world < 2:
        device = set_device(args.device)
    else:
        device = dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, world > 1)
    grid = phase_grid(rt, cfg)
    read = float(cfg['read_time'])
    sigma = rt.noise_scale()

    pool = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    cases = pool[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    shard, images_dir = args.out / f'shard_{rank:02d}.jsonl', args.out / f'img_{rank:02d}'
    images_dir.mkdir(exist_ok=True)

    mine = [case for position, case in enumerate(cases) if position % world == rank]
    for position, item in enumerate(mine):
        began = time.perf_counter()
        raw = item['prompt']
        seed = int(args.seed) + position * world + rank
        generator = torch.Generator(device=rt.device).manual_seed(seed)
        noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device,
                            generator=generator) * sigma
        velocity = NativeT2I(rt, raw, cfg)
        anchor = euler(noise.float(), segment_times(grid, 0.0, read), velocity)
        preview = preview_image(anchor, read, grid, NativeT2I(rt, raw, cfg))
        tag = f'{position * world + rank:03d}'
        save_image(images_dir / f'{tag}_preview.png', preview)

        review = review_reflection(rt, preview, raw, cfg)
        reflection = review['reflection']
        row = dict(index=position * world + rank, prompt_id=item.get('prompt_id'), prompt=raw,
                   skills=item.get('skills'), vqa_list=item.get('vqa_list'), seed=seed,
                   description=reflection['description'], match=reflection['match'],
                   instruction=reflection['instruction'], preserve=reflection['preserve'],
                   outcome=reflection['outcome'], raw_response=reflection['raw'][:2000],
                   seconds=time.perf_counter() - began)
        with shard.open('a') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(dict(rank=rank, index=row['index'], match=row['match'],
                              outcome=row['outcome']), ensure_ascii=False), flush=True)

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
        key = row.get('outcome') or 'unknown'
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(dict(cases=len(rows), outcomes=counts), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
