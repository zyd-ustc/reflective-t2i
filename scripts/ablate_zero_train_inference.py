"""Zero-training inference-side diagnostic ablation for isochronic_manifold_v1.

Tests principled inference hypotheses on a frozen stratified subset of triggered
GenEval2 cases (reusing exact cached Qwen30B reflections, baselines, and vanilla
isochronic scores) while sharing prefix z_{t0} and edit rollouts across arms:

  - arm1_cond_anchor  : Conditional native anchor H_0(z_{t0}, t0, prompt) vs empty
  - arm2_soft_mask_eta: cond_anchor + bbox-free self-emergent soft edit mask
                        (non-edit preservation + localized noise refresh on add/move)
  - arm3_mask_tmid    : arm2 + action-aware reprojection time t_out=0.48 on add/move
  - arm4_edit_cfg     : arm2 + edit_cfg_scale=1.75 during [0.36, 0.60] rollout
  - arm5_combo        : cond_anchor + soft_mask + localized eta + t_out=0.48 + edit_cfg=1.75
"""

import argparse
import copy
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("REFLECTIVE_WORKSPACE", ROOT.parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_MODEL = "/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT"
DEFAULT_SOURCE = str(WORKSPACE / "code/sensenova-u15-c35c83b")
DEFAULT_JUDGE_MODEL = "/data/zyd_workspace/models/Qwen3-VL-8B-Instruct"
DEFAULT_CKPT = "/data/zyd_workspace/isochronic_data_20261107/isochronic_manifold_400k_20261119/step_006001"
DEFAULT_CONFIG = str(ROOT / "configs/u15_isochronic_state/isochronic_manifold.json")
DEFAULT_SOURCE_BENCH = "/data/zyd_workspace/isochronic_data_20261107/bench_gv2_t2icomp64_400k_step006001_qwen30b_20261120"

ABLATION_ARMS = (
    "arm1_cond_anchor",
    "arm2_soft_mask_eta",
    "arm3_mask_tmid",
    "arm4_edit_cfg",
    "arm5_combo",
)


def _sync_device(device):
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _prefix_tuple(pref):
    if isinstance(pref, tuple):
        return pref
    return (pref.cache, pref.indexes, None)


def _load_source_records(source_bench: Path, benchmark: str = "geneval2"):
    cases_payload = json.loads((source_bench / "t2i_cases.json").read_text(encoding="utf-8"))
    case_by_id = {c["case_id"]: c for c in cases_payload["cases"] if c["benchmark"] == benchmark}

    gen_rows = {}
    for p in sorted((source_bench / "gen_shards").glob("gen_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("benchmark") == benchmark:
                    gen_rows[r["case_id"]] = r

    score_rows = {}
    for p in sorted((source_bench / "score_shards").glob("score_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("benchmark") == benchmark:
                    score_rows[r["case_id"]] = r

    return case_by_id, gen_rows, score_rows


def cmd_prepare(args):
    source_bench = Path(args.source_bench).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_by_id, gen_rows, score_rows = _load_source_records(source_bench, benchmark=args.benchmark)
    rng = random.Random(int(args.seed))

    trig_pass = []
    fails_by_action = defaultdict(list)
    for cid in sorted(case_by_id.keys()):
        gr = gen_rows.get(cid)
        sr = score_rows.get(cid)
        if not gr or not sr or gr.get("outcome") != "edit" or not gr.get("instruction"):
            continue
        b_score = sr["scores"]["baseline"]
        v_score = sr["scores"]["isochronic"]
        if b_score.get("correct") is None or v_score.get("correct") is None:
            continue
        entry = dict(
            case_by_id[cid],
            diagnosed_action=gr.get("diagnosed_action", "other"),
            instruction=gr["instruction"],
            baseline_score=b_score,
            vanilla_iso_score=v_score,
            baseline_image_abs=str((source_bench / gr["arms"]["baseline"]["image_rel"]).resolve()),
            vanilla_iso_image_abs=str((source_bench / gr["arms"]["isochronic"]["image_rel"]).resolve()),
            baseline_hf_grad=gr["arms"]["baseline"].get("hf_grad"),
            vanilla_hf_grad=gr["arms"]["isochronic"].get("hf_grad"),
            baseline_oob_frac=gr["arms"]["baseline"].get("oob_frac"),
            vanilla_oob_frac=gr["arms"]["isochronic"].get("oob_frac"),
        )
        if int(b_score["correct"]) == 1:
            trig_pass.append(entry)
        else:
            fails_by_action[entry["diagnosed_action"]].append(entry)

    selected = []
    if args.calibrate:
        # 8 diverse cases: 4 add (3 fail + 1 pass), 2 remove (1 fail + 1 pass), 2 attr (1 fail + 1 pass)
        pass_by_act = defaultdict(list)
        for e in trig_pass:
            pass_by_act[e["diagnosed_action"]].append(e)
        for act, nf, np_ in (("add", 3, 1), ("remove", 1, 1), ("attr", 1, 1)):
            selected.extend(fails_by_action[act][:nf])
            selected.extend(pass_by_act[act][:np_])
    else:
        # Include ALL trig_pass cases for exact Damage Rate, plus stratified trig_fail
        selected.extend(trig_pass)
        quotas = {
            "add": int(args.fail_add),
            "remove": int(args.fail_remove),
            "attr": int(args.fail_attr),
            "move": int(args.fail_move),
        }
        for act, q in quotas.items():
            pool = list(fails_by_action.get(act, []))
            rng.shuffle(pool)
            selected.extend(pool[:q])

    selected.sort(key=lambda x: x["case_id"])
    summary = Counter((e["diagnosed_action"], int(e["baseline_score"]["correct"])) for e in selected)
    payload = {
        "protocol": "zero_train_inference_ablation_v1",
        "source_bench": str(source_bench),
        "benchmark": args.benchmark,
        "seed": int(args.seed),
        "calibrate": bool(args.calibrate),
        "total_cases": len(selected),
        "counts_by_action_and_base_correct": {f"{k[0]}_base{k[1]}": v for k, v in sorted(summary.items())},
        "cases": selected,
    }
    out_file = out_dir / "ablation_cases.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(selected)} cases -> {out_file}", flush=True)
    print("Breakdown:", payload["counts_by_action_and_base_correct"], flush=True)


def cmd_gen_worker(args):
    from _time_probe_common import grad_mag, save_labeled_strip
    from qwen_latent_cot.u15_target_correction import GEOMETRY_PROTOCOL
    from qwen_latent_cot.u15_target_correction.accelerator import set_device
    from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.field import (
        NativeT2I,
        compute_self_emergent_edit_mask,
        euler,
        phase_grid,
        project_clean_to_native_manifold,
        segment_times,
    )
    from qwen_latent_cot.u15_target_correction.images import save_image
    from qwen_latent_cot.u15_target_correction.loading import load_native
    from qwen_latent_cot.u15_target_correction.model import build_editor

    torch.set_grad_enabled(False)
    device = set_device(args.device)
    cfg = load_config(args.config)
    t0 = float(cfg["read_time"])
    t_stop = float(cfg["stop_time"])
    t_mid = float(args.t_mid)
    mask_low = float(args.mask_low)
    mask_high = float(args.mask_high)
    mask_pool = int(args.mask_pool)
    mask_blur = float(args.mask_blur)
    eta_add = float(args.eta_add)
    edit_cfg_hi = float(args.edit_cfg_hi)

    out_dir = Path(args.out).resolve()
    payload = json.loads((out_dir / "ablation_cases.json").read_text(encoding="utf-8"))
    all_cases = payload["cases"]
    mine = [c for i, c in enumerate(all_cases) if i % args.world == args.rank]

    rt = load_native(args.model, args.source, cfg, device, distributed=False)
    editor = build_editor(rt, cfg).eval()
    ckpt_meta = load_stage_weights(
        args.checkpoint,
        editor,
        editor_identity(rt, cfg),
        allow_legacy_geometry=False,
    )
    if ckpt_meta.get("identity", {}).get("geometry") != GEOMETRY_PROTOCOL:
        raise RuntimeError("Checkpoint geometry mismatch")

    cfg_grid = copy.deepcopy(cfg)
    cfg_grid["phase_grid_times"] = sorted({t0, t_mid, t_stop})
    grid = phase_grid(rt, cfg_grid)
    sigma = float(rt.noise_scale())
    null_prefix = editor.empty_prefix()

    gen_dir = out_dir / "gen_shards"
    img_root = out_dir / "images"
    strip_dir = out_dir / "strips"
    gen_dir.mkdir(parents=True, exist_ok=True)
    img_root.mkdir(parents=True, exist_ok=True)
    strip_dir.mkdir(parents=True, exist_ok=True)

    shard_path = gen_dir / f"gen_rank{args.rank:02d}.jsonl"
    status_path = gen_dir / f"done_rank{args.rank:02d}.json"
    done_ids = set()
    if shard_path.exists():
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["case_id"])

    def _tensor_stats(z_raw):
        vis = z_raw.detach().float().clamp(-1.0, 1.0)
        oob = float(((z_raw < -1.0) | (z_raw > 1.0)).float().mean().item())
        hf = float(grad_mag(vis).mean().item())
        return round(oob, 5), round(hf, 5)

    def _rollout_clean(z_start, feedback_pref, cfg_scale):
        times_seg = segment_times(grid, t0, t_stop)
        q_stop = euler(
            z_start.detach().clone(),
            times_seg,
            lambda z, tau: editor.predict_velocity(
                z, tau, feedback_pref, empty_prefix=null_prefix, cfg_scale=cfg_scale
            ),
        )
        return editor.predict_guided_clean(
            q_stop, t_stop, feedback_pref, empty_prefix=null_prefix, cfg_scale=cfg_scale
        )

    for pos, case in enumerate(mine, 1):
        cid = case["case_id"]
        if cid in done_ids:
            continue
        t_case0 = time.perf_counter()
        prompt = case["prompt"]
        seed = int(case["seed"])
        action = case["diagnosed_action"]
        instruction = case["instruction"]
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cid)
        case_img_dir = img_root / safe_id
        case_img_dir.mkdir(parents=True, exist_ok=True)

        native = NativeT2I(rt, prompt, cfg)
        cond_anchor_pref = _prefix_tuple(native.positive)
        fb_pref = editor.feedback_prefix(instruction)

        # 1. Shared prefix 0 -> t0
        gen = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 3, rt.height, rt.width), device=device, generator=gen) * sigma
        z_t0 = euler(noise.float(), segment_times(grid, 0.0, t0), native)

        # 2. Extract conditional (and in calibrate mode, unconditional) clean anchor
        x_s_cond = editor.native_clean(z_t0.detach(), t0, cond_anchor_pref)
        eps_cond = (z_t0.float() - t0 * x_s_cond.float()) / (1.0 - t0)

        # Also advance native state to t_mid when action is add/move
        is_add_or_move = action in ("add", "move")
        z_tmid_native = (
            euler(z_t0.detach().clone(), segment_times(grid, t0, t_mid), native)
            if is_add_or_move
            else None
        )

        # 3. Shared edit rollouts: cfg=1.0 and cfg=edit_cfg_hi
        clean_cfg1 = _rollout_clean(z_t0, fb_pref, 1.0)
        clean_cfghi = _rollout_clean(z_t0, fb_pref, edit_cfg_hi)

        # 4. Self-emergent masks (bbox-free)
        mask_cfg1, mstats_cfg1 = compute_self_emergent_edit_mask(
            clean_cfg1,
            x_s_cond,
            low_thresh=mask_low,
            high_thresh=mask_high,
            pool_size=mask_pool,
            blur_sigma=mask_blur,
        )
        mask_cfghi, mstats_cfghi = compute_self_emergent_edit_mask(
            clean_cfghi,
            x_s_cond,
            low_thresh=mask_low,
            high_thresh=mask_high,
            pool_size=mask_pool,
            blur_sigma=mask_blur,
        )

        # Optional calibration check against unconditional anchor + vanilla reproduction
        calib_extra = {}
        if args.calibrate:
            x_s_uncond = editor.native_clean(z_t0.detach(), t0, null_prefix)
            _, mstats_uncond = compute_self_emergent_edit_mask(
                clean_cfg1,
                x_s_uncond,
                low_thresh=mask_low,
                high_thresh=mask_high,
                pool_size=mask_pool,
                blur_sigma=mask_blur,
            )
            eps_uncond = (z_t0.float() - t0 * x_s_uncond.float()) / (1.0 - t0)
            z_van = project_clean_to_native_manifold(clean_cfg1, eps_uncond, t0, sigma=sigma)
            raw_van = euler(z_van, segment_times(grid, t0, 1.0), native)
            van_oob, van_hf = _tensor_stats(raw_van)
            calib_extra = {
                "mask_stats_uncond_anchor": mstats_uncond,
                "vanilla_repro_oob": van_oob,
                "vanilla_repro_hf": van_hf,
                "cached_vanilla_hf": case.get("vanilla_hf_grad"),
            }

        eta_this = eta_add if is_add_or_move else 0.0
        t_out_mid = t_mid if is_add_or_move else t0
        preserve_mid = z_tmid_native if is_add_or_move else z_t0

        # Deterministic per-case fresh noise generator for localized noise refresh
        def _eta_gen(offset):
            return torch.Generator(device=device).manual_seed(seed * 31 + offset)

        arm_specs = {
            "arm1_cond_anchor": (
                project_clean_to_native_manifold(
                    clean_cfg1, eps_cond, t0, sigma=sigma, noise_refresh=0.0
                ),
                t0,
                {},
            ),
            "arm2_soft_mask_eta": (
                project_clean_to_native_manifold(
                    clean_cfg1,
                    eps_cond,
                    t0,
                    sigma=sigma,
                    noise_refresh_map=(eta_this * mask_cfg1) if eta_this > 0 else None,
                    preserve_state=z_t0,
                    edit_mask=mask_cfg1,
                    generator=_eta_gen(2),
                ),
                t0,
                mstats_cfg1,
            ),
            "arm3_mask_tmid": (
                project_clean_to_native_manifold(
                    clean_cfg1,
                    eps_cond,
                    t_out_mid,
                    sigma=sigma,
                    noise_refresh_map=(eta_this * mask_cfg1) if eta_this > 0 else None,
                    preserve_state=preserve_mid,
                    edit_mask=mask_cfg1,
                    generator=_eta_gen(3),
                ),
                t_out_mid,
                mstats_cfg1,
            ),
            "arm4_edit_cfg": (
                project_clean_to_native_manifold(
                    clean_cfghi,
                    eps_cond,
                    t0,
                    sigma=sigma,
                    noise_refresh=0.0,
                ),
                t0,
                mstats_cfghi,
            ),
            "arm5_combo": (
                project_clean_to_native_manifold(
                    clean_cfghi,
                    eps_cond,
                    t_out_mid,
                    sigma=sigma,
                    noise_refresh_map=(eta_this * mask_cfghi) if eta_this > 0 else None,
                    preserve_state=preserve_mid,
                    edit_mask=mask_cfghi,
                    generator=_eta_gen(5),
                ),
                t_out_mid,
                mstats_cfghi,
            ),
        }

        arms_out = {}
        strip_tiles = []
        for arm_name in ABLATION_ARMS:
            z_start, t_start, mstats = arm_specs[arm_name]
            _sync_device(device)
            t_s = time.perf_counter()
            raw = euler(z_start, segment_times(grid, t_start, 1.0), native)
            _sync_device(device)
            sec_tail = round(time.perf_counter() - t_s, 4)
            png_path = case_img_dir / f"{arm_name}.png"
            save_image(png_path, raw.clamp(-1.0, 1.0))
            oob, hf = _tensor_stats(raw)
            arms_out[arm_name] = dict(
                image_rel=str(png_path.relative_to(out_dir)),
                output_time=t_start,
                tail_seconds=sec_tail,
                oob_frac=oob,
                hf_grad=hf,
                mask_stats=mstats,
            )
            if args.calibrate or pos <= 10:
                strip_tiles.append((raw, arm_name))

        if strip_tiles:
            save_labeled_strip(
                [(x_s_cond, "x_S_cond(0.36)"), (clean_cfg1, "x_T_cfg1(0.60)"), (clean_cfghi, f"x_T_cfg{edit_cfg_hi}")]
                + strip_tiles,
                strip_dir / f"{safe_id}.jpg",
                tile_size=288,
                quality=88,
            )

        rec = dict(
            case_id=cid,
            benchmark=case["benchmark"],
            diagnosed_action=action,
            base_correct=int(case["baseline_score"]["correct"]),
            vanilla_correct=int(case["vanilla_iso_score"]["correct"]),
            instruction=instruction,
            prompt=prompt,
            seed=seed,
            mask_stats_cfg1=mstats_cfg1,
            mask_stats_cfghi=mstats_cfghi,
            arms=arms_out,
            calib_extra=calib_extra,
            wall_seconds=round(time.perf_counter() - t_case0, 3),
            rank=args.rank,
        )
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status_path.write_text(
            json.dumps({"rank": args.rank, "done": pos, "total": len(mine), "finished": False}) + "\n"
        )
        print(
            f"[gen {args.rank:02d}] ({pos}/{len(mine)}) {cid} ({action}, base={rec['base_correct']}) | "
            f"d95={mstats_cfg1['diff_p95']:.3f} m_mean={mstats_cfg1['mask_mean']:.3f} | "
            f"hf(a1={arms_out['arm1_cond_anchor']['hf_grad']:.4f}, a2={arms_out['arm2_soft_mask_eta']['hf_grad']:.4f}) | "
            f"{rec['wall_seconds']:.1f}s",
            flush=True,
        )

    status_path.write_text(
        json.dumps({"rank": args.rank, "done": len(mine), "total": len(mine), "finished": True}) + "\n"
    )
    print(f"[gen {args.rank:02d}] FINISHED {len(mine)} cases.", flush=True)


def cmd_score_worker(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_latent_cot.benchmarks.reference import t2i_protocols as proto

    torch.set_grad_enabled(False)
    if str(args.device).startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)

    out_dir = Path(args.out).resolve()
    payload = json.loads((out_dir / "ablation_cases.json").read_text(encoding="utf-8"))
    case_by_id = {c["case_id"]: c for c in payload["cases"]}

    gen_rows = {}
    for p in sorted((out_dir / "gen_shards").glob("gen_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                gen_rows[r["case_id"]] = r

    ordered_ids = [c["case_id"] for c in payload["cases"] if c["case_id"] in gen_rows]
    mine_ids = [cid for i, cid in enumerate(ordered_ids) if i % args.world == args.rank]

    score_dir = out_dir / "score_shards"
    score_dir.mkdir(parents=True, exist_ok=True)
    shard_path = score_dir / f"score_rank{args.rank:02d}.jsonl"
    status_path = score_dir / f"done_rank{args.rank:02d}.json"
    done_ids = set()
    if shard_path.exists():
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["case_id"])

    proc = AutoProcessor.from_pretrained(args.judge_model)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.judge_model, dtype=dtype).to(device).eval()

    gv2_sys = proto.QA_SYSTEM + " The answer field must contain exactly yes, no, unknown, or a single numeral from 0 to 9. Do not explain."

    def _ask_vlm(pil_img, user_text):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": gv2_sys}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt_text], images=[pil_img], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
        return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def _score_gv2(case, img_path: Path):
        with Image.open(img_path) as im:
            pil_img = im.convert("RGB")
        vqa_list = case.get("vqa_list") or []
        if not vqa_list:
            return {"status": "scored", "correct": 1, "qa_acc": 1.0, "answers": [], "raw": []}
        hits, answers, raws = [], [], []
        for q_text, expected_raw in vqa_list:
            raw = _ask_vlm(pil_img, q_text)
            raws.append(raw)
            try:
                ans = proto.interpret("geneval2", raw)
            except ValueError:
                ans = None
            expected = proto.normalize("geneval2", str(expected_raw))
            answers.append(ans)
            hits.append(int(ans is not None and expected is not None and ans == expected))
        ok = int(all(h == 1 for h in hits))
        return {
            "status": "scored",
            "correct": ok,
            "qa_acc": round(sum(hits) / len(hits), 4),
            "answers": answers,
            "raw": raws,
        }

    for pos, cid in enumerate(mine_ids, 1):
        if cid in done_ids:
            continue
        t0_s = time.perf_counter()
        case = case_by_id[cid]
        gr = gen_rows[cid]
        arm_scores = {
            "baseline": case["baseline_score"],
            "vanilla_iso": case["vanilla_iso_score"],
        }
        for arm_name in ABLATION_ARMS:
            arm_scores[arm_name] = _score_gv2(case, out_dir / gr["arms"][arm_name]["image_rel"])

        rec = dict(
            case_id=cid,
            diagnosed_action=case["diagnosed_action"],
            scores=arm_scores,
            score_seconds=round(time.perf_counter() - t0_s, 3),
            rank=args.rank,
        )
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status_path.write_text(
            json.dumps({"rank": args.rank, "done": pos, "total": len(mine_ids), "finished": False}) + "\n"
        )
        summary_str = " ".join(
            f"{a[:4]}={arm_scores[a]['correct']}" for a in ("baseline", "vanilla_iso", *ABLATION_ARMS)
        )
        print(f"[score {args.rank:02d}] ({pos}/{len(mine_ids)}) {cid} ({case['diagnosed_action']}) | {summary_str}", flush=True)

    status_path.write_text(
        json.dumps({"rank": args.rank, "done": len(mine_ids), "total": len(mine_ids), "finished": True}) + "\n"
    )
    print(f"[score {args.rank:02d}] FINISHED {len(mine_ids)} cases.", flush=True)


def cmd_merge(args):
    out_dir = Path(args.out).resolve()
    payload = json.loads((out_dir / "ablation_cases.json").read_text(encoding="utf-8"))
    case_by_id = {c["case_id"]: c for c in payload["cases"]}

    gen_rows = {}
    for p in sorted((out_dir / "gen_shards").glob("gen_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                gen_rows[r["case_id"]] = r

    score_rows = {}
    for p in sorted((out_dir / "score_shards").glob("score_rank*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                score_rows[r["case_id"]] = r

    all_arms = ("baseline", "vanilla_iso", *ABLATION_ARMS)
    valid_ids = [c["case_id"] for c in payload["cases"] if c["case_id"] in score_rows and c["case_id"] in gen_rows]

    def _summarize(ids):
        out = {}
        for arm in all_arms:
            n = len(ids)
            acc = sum(int(score_rows[cid]["scores"][arm]["correct"]) for cid in ids) / max(n, 1)
            qa = sum(float(score_rows[cid]["scores"][arm]["qa_acc"]) for cid in ids) / max(n, 1)
            tf = [cid for cid in ids if int(score_rows[cid]["scores"]["baseline"]["correct"]) == 0]
            tp = [cid for cid in ids if int(score_rows[cid]["scores"]["baseline"]["correct"]) == 1]
            fixed = sum(int(score_rows[cid]["scores"][arm]["correct"]) == 1 for cid in tf)
            dmg = sum(int(score_rows[cid]["scores"][arm]["correct"]) == 0 for cid in tp)
            if arm == "baseline":
                hfs = [case_by_id[cid]["baseline_hf_grad"] for cid in ids if case_by_id[cid].get("baseline_hf_grad") is not None]
                oobs = [case_by_id[cid]["baseline_oob_frac"] for cid in ids if case_by_id[cid].get("baseline_oob_frac") is not None]
            elif arm == "vanilla_iso":
                hfs = [case_by_id[cid]["vanilla_hf_grad"] for cid in ids if case_by_id[cid].get("vanilla_hf_grad") is not None]
                oobs = [case_by_id[cid]["vanilla_oob_frac"] for cid in ids if case_by_id[cid].get("vanilla_oob_frac") is not None]
            else:
                hfs = [gen_rows[cid]["arms"][arm]["hf_grad"] for cid in ids]
                oobs = [gen_rows[cid]["arms"][arm]["oob_frac"] for cid in ids]
            out[arm] = dict(
                n=n,
                acc=round(acc, 4),
                qa_acc=round(qa, 4),
                fixed=fixed,
                trig_fail=len(tf),
                fix_rate=round(fixed / max(len(tf), 1), 4),
                damaged=dmg,
                trig_pass=len(tp),
                damage_rate=round(dmg / max(len(tp), 1), 4),
                net_fixed=fixed - dmg,
                mean_hf=round(sum(hfs) / max(len(hfs), 1), 5) if hfs else None,
                mean_oob=round(sum(oobs) / max(len(oobs), 1), 5) if oobs else None,
            )
        return out

    overall = _summarize(valid_ids)
    by_action = {}
    for act in ("add", "remove", "attr", "move"):
        act_ids = [cid for cid in valid_ids if case_by_id[cid]["diagnosed_action"] == act]
        if act_ids:
            by_action[act] = _summarize(act_ids)

    # Also compute mask telemetry means
    m_cfg1 = [gen_rows[cid]["mask_stats_cfg1"] for cid in valid_ids]
    mask_summary = {
        k: round(sum(m[k] for m in m_cfg1) / max(len(m_cfg1), 1), 4)
        for k in ("diff_p50", "diff_p90", "diff_p95", "diff_max", "mask_mean", "mask_max")
    }

    report = {
        "protocol": "zero_train_inference_ablation_v1",
        "expected": len(payload["cases"]),
        "scored": len(valid_ids),
        "complete": len(valid_ids) == len(payload["cases"]),
        "overall": overall,
        "by_action": by_action,
        "mask_summary_cfg1": mask_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# Zero-Training Inference Ablation Summary (`{len(valid_ids)}/{len(payload['cases'])}` GenEval2 Triggered Cases)",
        "",
        "## Overall",
        "| Arm | Exact Acc ↑ | Atomic QA Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF | Mean OOB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in all_arms:
        s = overall[arm]
        lines.append(
            f"| `{arm}` | {s['acc']:.4f} | {s['qa_acc']:.4f} | "
            f"{s['fixed']}/{s['trig_fail']} ({s['fix_rate']:.1%}) | "
            f"{s['damaged']}/{s['trig_pass']} ({s['damage_rate']:.1%}) | "
            f"{s['net_fixed']:+d} | {s['mean_hf']} | {s['mean_oob']} |"
        )
    for act, stats in by_action.items():
        lines.extend([
            "",
            f"## Action: `{act}` (n={stats['baseline']['n']})",
            "| Arm | Exact Acc ↑ | Atomic QA Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for arm in all_arms:
            s = stats[arm]
            lines.append(
                f"| `{arm}` | {s['acc']:.4f} | {s['qa_acc']:.4f} | "
                f"{s['fixed']}/{s['trig_fail']} ({s['fix_rate']:.1%}) | "
                f"{s['damaged']}/{s['trig_pass']} ({s['damage_rate']:.1%}) | "
                f"{s['net_fixed']:+d} | {s['mean_hf']} |"
            )
    lines.extend(["", "## Self-Emergent Mask Stats (cfg=1.0)", f"`{json.dumps(mask_summary)}`", ""])
    md = "\n".join(lines)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--source-bench", default=DEFAULT_SOURCE_BENCH)
    sp.add_argument("--benchmark", default="geneval2")
    sp.add_argument("--out", required=True)
    sp.add_argument("--seed", type=int, default=20261122)
    sp.add_argument("--calibrate", action="store_true")
    sp.add_argument("--fail-add", type=int, default=85)
    sp.add_argument("--fail-remove", type=int, default=25)
    sp.add_argument("--fail-attr", type=int, default=20)
    sp.add_argument("--fail-move", type=int, default=5)
    sp.set_defaults(func=cmd_prepare)

    sg = sub.add_parser("gen-worker")
    sg.add_argument("--out", required=True)
    sg.add_argument("--rank", type=int, required=True)
    sg.add_argument("--world", type=int, required=True)
    sg.add_argument("--device", default="npu:0")
    sg.add_argument("--model", default=DEFAULT_MODEL)
    sg.add_argument("--source", default=DEFAULT_SOURCE)
    sg.add_argument("--checkpoint", default=DEFAULT_CKPT)
    sg.add_argument("--config", default=DEFAULT_CONFIG)
    sg.add_argument("--calibrate", action="store_true")
    sg.add_argument("--t-mid", type=float, default=0.48)
    sg.add_argument("--mask-low", type=float, default=0.04)
    sg.add_argument("--mask-high", type=float, default=0.18)
    sg.add_argument("--mask-pool", type=int, default=16)
    sg.add_argument("--mask-blur", type=float, default=1.2)
    sg.add_argument("--eta-add", type=float, default=0.50)
    sg.add_argument("--edit-cfg-hi", type=float, default=1.75)
    sg.set_defaults(func=cmd_gen_worker)

    ss = sub.add_parser("score-worker")
    ss.add_argument("--out", required=True)
    ss.add_argument("--rank", type=int, required=True)
    ss.add_argument("--world", type=int, required=True)
    ss.add_argument("--device", default="npu:0")
    ss.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ss.set_defaults(func=cmd_score_worker)

    sm = sub.add_parser("merge")
    sm.add_argument("--out", required=True)
    sm.set_defaults(func=cmd_merge)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
