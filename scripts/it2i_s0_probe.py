"""Zero-training IT2I-prefix isochronic probe at fixed t=read_time.

Compares three conditioning arms at s=0 (and optional 8-step s-Euler) on
family-stratified train/val records:

1. ``t2i_text``      Current text-only T2I prefix ("Edit request: <e>" vs None).
2. ``it2i_source``   Native IT2I prefix using clean source x_a + raw instruction e.
3. ``it2i_preview``  Native IT2I prefix using lookahead preview P(a_t) + instruction e.

Reports full/low-frequency cosine, displacement ratio, and projected progress
ratio per split and per edit family (count/spatial/attribute/replace/other).
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_latent_cot.u15_target_correction.accelerator import set_device
from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
from qwen_latent_cot.u15_target_correction.config import load_config
from qwen_latent_cot.u15_target_correction.field import NativeT2I, phase_grid, preview_image
from qwen_latent_cot.u15_target_correction.images import load_canvas
from qwen_latent_cot.u15_target_correction.loading import load_native
from qwen_latent_cot.u15_target_correction.model import StateEditor
from qwen_latent_cot.u15_target_correction.runtime import autocast


def _family(instruction):
    text = (instruction or "").lower()
    if text.startswith(("add ", "remove ")):
        return "count"
    if text.startswith(("rearrange", "swap", "move")):
        return "spatial"
    if text.startswith("replace"):
        return "replace"
    if text.startswith("change"):
        return "attribute"
    return "other"


def _pick_stratified(records, per_split, seed):
    buckets = {}
    for r in records:
        buckets.setdefault(_family(r.get("edit_instruction")), []).append(r)
    rng = random.Random(seed)
    for items in buckets.values():
        items.sort(key=lambda x: x["uid"])
        rng.shuffle(items)
    chosen = []
    while len(chosen) < per_split and any(buckets.values()):
        for fam in sorted(buckets):
            if buckets[fam] and len(chosen) < per_split:
                chosen.append(buckets[fam].pop())
    return chosen


def _lowfreq(tensor, pool=16):
    return F.avg_pool2d(tensor.detach().float(), kernel_size=pool, stride=pool).reshape(-1)


def _metrics(disp, target_disp):
    a_full, b_full = disp.detach().float().reshape(-1), target_disp.detach().float().reshape(-1)
    a_low, b_low = _lowfreq(disp), _lowfreq(target_disp)
    b_full_norm = float(b_full.norm()) + 1e-8
    b_low_norm = float(b_low.norm()) + 1e-8
    return dict(
        cos_full=float(F.cosine_similarity(a_full, b_full, dim=0)),
        cos_low=float(F.cosine_similarity(a_low, b_low, dim=0)),
        disp_ratio_full=float(a_full.norm()) / b_full_norm,
        disp_ratio_low=float(a_low.norm()) / b_low_norm,
        proj_ratio_full=float(torch.dot(a_full, b_full)) / (b_full_norm ** 2),
        proj_ratio_low=float(torch.dot(a_low, b_low)) / (b_low_norm ** 2),
    )


def _it2i_pixels(rt, image):
    m = rt.model
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    rgb = ((image.detach().float().clamp(-1, 1) + 1) / 2 - mean) / std
    gh, gw = rt.height // m.patch_size, rt.width // m.patch_size
    grid = torch.tensor([[gh, gw]], device=rt.device)
    pixels = (
        m.patchify(rgb, m.patch_size, channel_first=True)
        .reshape(gh * gw, -1)
        .to(m.vision_model.embeddings.patch_embedding.weight.dtype)
    )
    return pixels, grid


def _it2i_prefix(rt, vit, grid, instruction, *, image_only):
    from sensenova_u1.models.neo_unify.modeling_qwen3 import create_block_causal_mask
    from sensenova_u1.models.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

    m, tok = rt.model, rt.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    m.img_start_token_id = tok.convert_tokens_to_ids("<img>")
    prompt = "<image>" if image_only else "<image>\n" + instruction
    if image_only:
        query = m._build_t2i_query(prompt, append_text="<img>")
    else:
        query = m._build_t2i_query(
            prompt, system_message=SYSTEM_MESSAGE_FOR_GEN, append_text="<think>\n\n</think>\n\n<img>"
        )
    count = int(grid[0, 0] * grid[0, 1] * m.downsample_ratio ** 2)
    tokens = "<img>" + "<IMG_CONTEXT>" * count + "</img>"
    query = query.replace("<image>", tokens, 1)

    encoded = tok(query, return_tensors="pt")
    input_ids = encoded["input_ids"].to(rt.device)
    indexes = m.get_thw_indexes(input_ids[0], grid)
    attention_mask = {"full_attention": create_block_causal_mask(indexes[0])}
    embeds = m.language_model.get_input_embeddings()(input_ids)
    batch, length, channels = embeds.shape
    selected = input_ids.reshape(batch * length) == m.img_context_token_id
    visual = vit.reshape(-1, channels)
    if int(selected.sum()) != visual.shape[0]:
        raise ValueError("Condition image tokens differ from the generation canvas")
    embeds = embeds.reshape(batch * length, channels).clone()
    embeds[selected] = visual.to(embeds.dtype)
    embeds = embeds.reshape(batch, length, channels)

    with autocast(rt.device):
        cache, _ = m._it2i_prefix_forward(embeds, indexes, attention_mask)
        gen_index = m._build_t2i_image_indexes(*rt.grid, int(indexes[0].max()) + 1, device=rt.device)
    return (cache, gen_index, None)


def _it2i_pair(rt, image, instruction):
    pixels, grid = _it2i_pixels(rt, image)
    with autocast(rt.device):
        vit = rt.model.extract_feature(pixels, grid_hw=grid)
    return (
        _it2i_prefix(rt, vit, grid, "", image_only=True),
        _it2i_prefix(rt, vit, grid, instruction, image_only=False),
    )


def _native_it2i_rollout(editor, anchor, read, image_prefix, joint_prefix, steps, cfg_scale):
    """Zero-training isochronic Euler rollout using frozen native IT2I prefixes."""
    state = anchor.float()
    base = editor.image_clean(state, read, image_prefix)
    for index in range(steps):
        if cfg_scale == 1.0:
            pred_cond = editor.image_clean(state, read, joint_prefix)
            vel = (pred_cond - base) * float(read)
        else:
            pred_cond = editor.image_clean(state, read, joint_prefix)
            pred_uncond = editor.image_clean(state, read, image_prefix)
            guided = pred_uncond + float(cfg_scale) * (pred_cond - pred_uncond)
            vel = (guided - base) * float(read)
        state = state + vel / steps
    return state - anchor.float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for t2i_text arm comparison")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-split", type=int, default=65)
    parser.add_argument("--rollout-steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    from qwen_latent_cot.u15_target_correction import distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = set_device(args.device) if world < 2 else dist.initialize(args.device)

    torch.set_grad_enabled(False)
    cfg = load_config(args.config)
    rt = load_native(args.model, args.source, cfg, device, world > 1)
    editor = StateEditor(rt, cfg).eval()
    if args.checkpoint:
        load_stage_weights(args.checkpoint, editor, editor_identity(rt, cfg))

    grid = phase_grid(rt, cfg)
    read = float(cfg["read_time"])
    sigma = rt.noise_scale()
    args.out.mkdir(parents=True, exist_ok=True)
    shard = args.out / f"shard_{rank:02d}.jsonl"

    records = json.loads(args.manifest.read_text())["records"]
    for split in ("train", "val"):
        pool = _pick_stratified([r for r in records if r["split"] == split], args.per_split, args.seed)
        for position, record in enumerate(pool):
            if position % world != rank:
                continue
            began = time.perf_counter()
            source, _, _ = load_canvas(Path(record["source_image"]), rt.width, rt.height, rt.device)
            target, _, _ = load_canvas(Path(record["target_image"]), rt.width, rt.height, rt.device)
            generator = torch.Generator(device=rt.device).manual_seed(abs(hash(record["uid"])) % (2 ** 31))
            noise = torch.randn((1, 3, rt.height, rt.width), device=rt.device, generator=generator) * sigma
            anchor = read * source + (1 - read) * noise
            true_disp = read * target + (1 - read) * noise - anchor
            instruction = record["edit_instruction"]
            family = _family(instruction)

            # Arm 1: Current text-only T2I prefix
            t2i_pref = editor.prefixes(instruction)
            t2i_base = editor.image_clean(anchor, read, t2i_pref["image"])
            t2i_s0 = editor.state_velocity(anchor, read, 0.0, t2i_pref["joint"], t2i_base)

            # Arm 2: Zero-training native IT2I with clean source x_a
            src_img_pref, src_joint_pref = _it2i_pair(rt, source, instruction)
            src_base = editor.image_clean(anchor, read, src_img_pref)
            src_pred = editor.image_clean(anchor, read, src_joint_pref)
            it2i_src_s0 = (src_pred - src_base) * read

            # Arm 3: Zero-training native IT2I with lookahead preview P(a_t)
            preview = preview_image(anchor, read, grid, NativeT2I(rt, None, cfg))
            prv_img_pref, prv_joint_pref = _it2i_pair(rt, preview, instruction)
            prv_base = editor.image_clean(anchor, read, prv_img_pref)
            prv_pred = editor.image_clean(anchor, read, prv_joint_pref)
            it2i_prv_s0 = (prv_pred - prv_base) * read

            arms = dict(
                t2i_text_s0=_metrics(t2i_s0, true_disp),
                it2i_source_s0=_metrics(it2i_src_s0, true_disp),
                it2i_preview_s0=_metrics(it2i_prv_s0, true_disp),
            )
            if args.rollout_steps > 0:
                it2i_prv_end = _native_it2i_rollout(
                    editor, anchor, read, prv_img_pref, prv_joint_pref, args.rollout_steps, args.cfg_scale
                )
                arms[f"it2i_preview_step{args.rollout_steps}_w{args.cfg_scale:g}"] = _metrics(it2i_prv_end, true_disp)

            row = dict(
                split=split, uid=record["uid"], family=family, instruction=instruction,
                arms=arms, seconds=time.perf_counter() - began,
            )
            with shard.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(dict(
                split=split, uid=record["uid"], family=family,
                t2i_cos=round(arms["t2i_text_s0"]["cos_low"], 3),
                src_it2i_cos=round(arms["it2i_source_s0"]["cos_low"], 3),
                prv_it2i_cos=round(arms["it2i_preview_s0"]["cos_low"], 3),
            ), ensure_ascii=False), flush=True)

    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    all_rows = []
    for path in sorted(args.out.glob("shard_*.jsonl")):
        all_rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    (args.out / "rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=1))

    arm_names = list(all_rows[0]["arms"].keys()) if all_rows else []
    metric_keys = ("cos_full", "cos_low", "disp_ratio_low", "proj_ratio_low")

    def _agg(items):
        if not items:
            return {}
        return {
            arm: {k: sum(r["arms"][arm][k] for r in items) / len(items) for k in metric_keys}
            for arm in arm_names
        }

    summary = {}
    for split in ("train", "val"):
        sub = [r for r in all_rows if r["split"] == split]
        if not sub:
            continue
        summary[split] = dict(
            cases=len(sub),
            overall=_agg(sub),
            by_family={fam: _agg([r for r in sub if r["family"] == fam]) for fam in sorted({r["family"] for r in sub})},
        )
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
