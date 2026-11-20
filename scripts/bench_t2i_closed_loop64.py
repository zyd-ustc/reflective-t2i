"""64-NPU End-to-End Real T2I Closed-Loop Benchmark for the Strict Isochronic Manifold Operator.

Two-phase decoupled execution on 4x16 Ascend 910C NPUs (64 chips total):
  1. prepare       : Freeze full GenEval (553) + GenEval++ (280) = 833 T2I cases from frozen manifests.
  2. gen-worker    : Run SenseNova-U1.5 shared prefix (0 -> t0) + Breakthrough-A multi-step lookahead
                     preview & gated UND reflection + 2 arms:
                       - baseline   : Native T2I G1 (0 -> 1.0, cfg_interval=[0.0, 0.80])
                       - isochronic : Strict isochronic manifold transition z_{t0} -> z_hat_{t0}
                                      (internal rollout [t0, stop_time]) + Frozen Native G1 tail (t0 -> 1.0)
  3. score-worker  : Score all generated cases with independent Qwen3-VL-8B-Instruct using the exact
                     frozen GenEval per-clause QA logic and GenEval++ checklist rubric.
  4. launch-gen64 / launch-score64 : Preflight 4 nodes (64 NPUs) and spawn 64 detached workers.
  5. merge         : Verify 833/833 completeness, compute category macro-means, paired bootstrap 95% CIs,
                     Breakthrough-A gate telemetry, and build HTML gallery + summary.json / summary.md.
"""

import argparse
import copy
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_U15_PY = "/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python"
DEFAULT_QWEN_PY = "/data/zyd_workspace/bagel-LatentCoT/envs/qwen3vl-score/bin/python"
DEFAULT_MODEL = "/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT"
DEFAULT_SOURCE = "/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b"
DEFAULT_JUDGE_MODEL = "/data/zyd_workspace/models/Qwen3-VL-8B-Instruct"
DEFAULT_CONFIG = str(ROOT / "configs/u15_isochronic_state/isochronic_manifold.json")
DEFAULT_GENEVAL = (
    "/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/"
    "0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/geneval.jsonl"
)
DEFAULT_GENEVAL_PP = (
    "/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/"
    "0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/geneval_pp.jsonl"
)
DEFAULT_GENEVAL2 = "/data/zyd_workspace/bagel-LatentCoT/code/anchor-search-20261104/data/geneval2_800.jsonl"
DEFAULT_T2I_COMPBENCH_PP = (
    "/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/"
    "0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/t2i_compbench_pp.jsonl"
)

PER_NODE = 16
ARMS = ("baseline", "isochronic")
GENEVAL_CATS = ("single_object", "two_object", "counting", "colors", "position", "color_attr")


def _load_ref_module(name, rel_path):
    full_path = ROOT / rel_path
    common_dir = ROOT / "qwen_latent_cot/benchmarks/reference/scoring_sources/bench_tools/common"
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def classify_diagnosed_action(instruction):
    """Classify UND's spontaneous edit instruction into add / remove / move / attr / other."""
    from qwen_latent_cot.u15_target_correction.data import classify_semantic_task

    text = " ".join(str(instruction or "").split())
    if not text:
        return "none"
    sem, direction = classify_semantic_task(text.lower(), "")
    if sem == "count" and direction in ("add", "remove"):
        return direction
    if sem == "spatial":
        return "move"
    if sem == "attribute":
        return "attr"
    low = text.lower()
    if low.startswith("add "):
        return "add"
    if low.startswith("remove "):
        return "remove"
    if low.startswith(("move ", "relocate ", "reposition ", "swap ")):
        return "move"
    if low.startswith(("change ", "replace ", "make ")):
        return "attr"
    return "other"


def build_t2i_cases(geneval_jsonl=None, geneval_pp_jsonl=None, geneval2_jsonl=None, t2i_compbench_jsonl=None):
    """Build deterministic frozen benchmark list without touching historical images."""
    items = []
    if geneval_jsonl:
        gv_rows = [json.loads(l) for l in Path(geneval_jsonl).read_text().splitlines() if l.strip()]
        for idx, r in enumerate(gv_rows):
            prompt = r.get("geneval_prompt") or r.get("instruction") or r.get("original_prompt")
            if not prompt or not str(prompt).strip():
                raise ValueError(f"Empty prompt in GenEval row {r.get('record_id')}")
            cat = str(r.get("geneval_tag") or r.get("category") or "other")
            items.append(
                dict(
                    case_id=f"geneval:{r['record_id']}",
                    benchmark="geneval",
                    idx_in_bench=idx,
                    record_id=str(r["record_id"]),
                    category=cat,
                    prompt=str(prompt).strip(),
                    seed=int(r["seed"]),
                    geneval_include=r.get("geneval_include", []),
                    geneval_exclude=r.get("geneval_exclude", []),
                )
            )
    if geneval_pp_jsonl:
        pp_rows = [json.loads(l) for l in Path(geneval_pp_jsonl).read_text().splitlines() if l.strip()]
        for idx, r in enumerate(pp_rows):
            prompt = r.get("required_prompt") or r.get("instruction")
            if not prompt or not str(prompt).strip():
                raise ValueError(f"Empty prompt in GenEval++ row {r.get('record_id')}")
            if r.get("instruction") and str(r["instruction"]).strip() != str(prompt).strip():
                raise ValueError(f"GenEval++ prompt mismatch in {r.get('record_id')}")
            cat = str(r.get("category") or r.get("benchmark_metadata", {}).get("tag") or "other")
            items.append(
                dict(
                    case_id=f"geneval_pp:{r['record_id']}",
                    benchmark="geneval_pp",
                    idx_in_bench=idx,
                    record_id=str(r["record_id"]),
                    category=cat,
                    prompt=str(prompt).strip(),
                    seed=int(r["seed"]),
                    benchmark_metadata=r["benchmark_metadata"],
                )
            )
    if geneval2_jsonl:
        gv2_rows = [json.loads(l) for l in Path(geneval2_jsonl).read_text().splitlines() if l.strip()]
        for idx, r in enumerate(gv2_rows):
            prompt = r.get("prompt") or r.get("raw_prompt") or r.get("instruction")
            if not prompt or not str(prompt).strip():
                raise ValueError(f"Empty prompt in GenEval2 row {idx}")
            rid = str(r.get("prompt_id") or r.get("record_id") or f"geneval2_{idx:04d}")
            cat = str(r.get("category") or (f"atoms_{r['bucket']}" if r.get("bucket") is not None else "all"))
            items.append(
                dict(
                    case_id=f"geneval2:{rid}",
                    benchmark="geneval2",
                    idx_in_bench=idx,
                    record_id=rid,
                    category=cat,
                    prompt=str(prompt).strip(),
                    seed=int(r.get("seed", 20260900 + idx)),
                    vqa_list=r["vqa_list"],
                    skills=r.get("skills", []),
                )
            )
    if t2i_compbench_jsonl:
        tc_rows = [json.loads(l) for l in Path(t2i_compbench_jsonl).read_text().splitlines() if l.strip()]
        for idx, r in enumerate(tc_rows):
            prompt = r.get("required_prompt") or r.get("instruction")
            if not prompt or not str(prompt).strip():
                raise ValueError(f"Empty prompt in T2I-CompBench++ row {r.get('record_id')}")
            cat = str(r.get("category") or r.get("benchmark_metadata", {}).get("category") or "other")
            items.append(
                dict(
                    case_id=f"t2i_compbench_pp:{r['record_id']}",
                    benchmark="t2i_compbench_pp",
                    idx_in_bench=idx,
                    record_id=str(r["record_id"]),
                    category=cat,
                    prompt=str(prompt).strip(),
                    seed=int(r["seed"]),
                    benchmark_metadata=r["benchmark_metadata"],
                )
            )
    counts = Counter(x["benchmark"] for x in items)
    return {
        "protocol": "t2i_isochronic_closed_loop64_v2",
        "benchmarks": dict(counts),
        "total_cases": len(items),
        "cases": items,
    }


def cmd_prepare(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_t2i_cases(
        args.geneval if not args.skip_geneval else None,
        args.geneval_pp if not args.skip_geneval_pp else None,
        args.geneval2 if getattr(args, "geneval2", None) else None,
        args.t2i_compbench if getattr(args, "t2i_compbench", None) else None,
    )
    out_file = out_dir / "t2i_cases.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Prepared {payload['total_cases']} cases ({payload['benchmarks']}) -> {out_file}", flush=True)


def _sync_device(device):
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _apply_runtime_overrides(cfg, args):
    run_cfg = copy.deepcopy(cfg)
    if getattr(args, "stop_time", None) is not None:
        run_cfg["stop_time"] = float(args.stop_time)
    if getattr(args, "lookahead_steps", None) is not None:
        run_cfg["lookahead_steps"] = int(args.lookahead_steps)
    if getattr(args, "edit_cfg_scale", None) is not None:
        run_cfg["edit_cfg_scale"] = float(args.edit_cfg_scale)
    if getattr(args, "no_recheck_add", False):
        run_cfg["recheck_add_on_preview"] = False
    if getattr(args, "no_consistency_gate", False):
        run_cfg["gate_contradictory_edits"] = False
    from qwen_latent_cot.u15_target_correction.config import validate_config

    validate_config(run_cfg)
    return run_cfg


def cmd_gen_worker(args):
    from _time_probe_common import grad_mag, save_labeled_strip
    from qwen_latent_cot.u15_target_correction import GEOMETRY_PROTOCOL
    from qwen_latent_cot.u15_target_correction.accelerator import set_device
    from qwen_latent_cot.u15_target_correction.checkpoint import editor_identity, load_stage_weights
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.diagnosis import review_reflection
    from qwen_latent_cot.u15_target_correction.field import (
        NativeT2I,
        euler,
        isochronic_manifold_transition,
        phase_grid,
        preview_image,
        segment_times,
    )
    from qwen_latent_cot.u15_target_correction.images import save_image
    from qwen_latent_cot.u15_target_correction.loading import load_native
    from qwen_latent_cot.u15_target_correction.model import build_editor

    torch.set_grad_enabled(False)
    device = set_device(args.device)
    cfg = _apply_runtime_overrides(load_config(args.config), args)
    t0 = float(cfg["read_time"])
    stop_time = float(cfg["stop_time"])
    lookahead = int(cfg["lookahead_steps"])
    edit_cfg_scale = float(cfg["edit_cfg_scale"])

    out_dir = Path(args.out).resolve()
    payload = json.loads((out_dir / "t2i_cases.json").read_text())
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

    grid = phase_grid(rt, cfg)
    sigma = float(rt.noise_scale())
    null_prefix = editor.empty_prefix()

    reflection_cache = {}
    if getattr(args, "reflection_cache", None):
        for line in Path(args.reflection_cache).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rr = json.loads(line)
                reflection_cache[rr["case_id"]] = rr

    prev_gen_rows = {}
    prev_dir = Path(args.reuse_baseline_from).resolve() if getattr(args, "reuse_baseline_from", None) else None
    if prev_dir is not None:
        for p in sorted((prev_dir / "gen_shards").glob("gen_rank*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    pr = json.loads(line)
                    prev_gen_rows[pr["case_id"]] = pr

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
        for line in shard_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["case_id"])

    def _tensor_stats(z_raw):
        vis = z_raw.detach().float().clamp(-1.0, 1.0)
        oob = float(((z_raw < -1.0) | (z_raw > 1.0)).float().mean().item())
        hf = float(grad_mag(vis).mean().item())
        return round(oob, 5), round(hf, 5)

    for pos, case in enumerate(mine, 1):
        cid = case["case_id"]
        if cid in done_ids:
            continue
        t_case0 = time.perf_counter()
        prompt = case["prompt"]
        seed = int(case["seed"])
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", cid)
        case_img_dir = img_root / case["benchmark"] / safe_id
        case_img_dir.mkdir(parents=True, exist_ok=True)

        native = NativeT2I(rt, prompt, cfg)
        prev_rec = prev_gen_rows.get(cid)
        cached_ref = reflection_cache.get(cid)

        if cached_ref is not None:
            review = cached_ref
            outcome = str(review["outcome"])
            instruction = " ".join(str(review.get("instruction") or "").split())
            diag_action = classify_diagnosed_action(instruction) if outcome == "edit" else "none"
            sec_reflect = 0.0
            nfe_preview = 0
        else:
            review = None

        base_png = case_img_dir / "baseline.png"
        prev_png = case_img_dir / "preview.png"
        base_rel = str(base_png.relative_to(out_dir))

        if prev_rec is not None and cached_ref is not None:
            prev_base_src = prev_dir / prev_rec["arms"]["baseline"]["image_rel"]
            prev_prev_src = prev_base_src.parent / "preview.png"
            if not base_png.exists():
                os.symlink(prev_base_src, base_png)
            if prev_prev_src.exists() and not prev_png.exists():
                os.symlink(prev_prev_src, prev_png)
            base_arm = prev_rec["arms"]["baseline"]
            base_oob = base_arm["oob_frac"]
            base_hf = base_arm["hf_grad"]
            nfe_prefix = int(round(base_arm["total_nfe"] * t0))
            nfe_base_tail = int(base_arm["total_nfe"]) - nfe_prefix
            sec_prefix = round(float(base_arm["total_seconds"]) * t0, 4)
            sec_base_tail = round(float(base_arm["total_seconds"]) - sec_prefix, 4)
            arms_info = {"baseline": dict(base_arm, image_rel=base_rel)}
            if outcome == "edit" and instruction:
                gen = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn((1, 3, rt.height, rt.width), device=device, generator=gen) * sigma
                _sync_device(device)
                t0_s = time.perf_counter()
                native.calls = 0
                z_t0 = euler(noise.float(), segment_times(grid, 0.0, t0), native)
                _sync_device(device)
                sec_prefix = time.perf_counter() - t0_s
                nfe_prefix = int(native.calls)
                preview = Image.open(prev_prev_src).convert("RGB") if prev_prev_src.exists() else None
                raw_base = Image.open(prev_base_src).convert("RGB")
                # Convert PIL to [-1, 1] tensor only for strip if needed
                import torchvision.transforms.functional as TF
                preview_t = TF.to_tensor(preview).unsqueeze(0).to(device) * 2.0 - 1.0 if preview is not None else None
                raw_base_t = TF.to_tensor(raw_base).unsqueeze(0).to(device) * 2.0 - 1.0
                preview = preview_t if preview_t is not None else raw_base_t
                raw_base = raw_base_t
        else:
            gen = torch.Generator(device=device).manual_seed(seed)
            noise = torch.randn((1, 3, rt.height, rt.width), device=device, generator=gen) * sigma

            _sync_device(device)
            t0_s = time.perf_counter()
            native.calls = 0
            z_t0 = euler(noise.float(), segment_times(grid, 0.0, t0), native)
            _sync_device(device)
            sec_prefix = time.perf_counter() - t0_s
            nfe_prefix = int(native.calls)

            _sync_device(device)
            t0_s = time.perf_counter()
            native.calls = 0
            preview = preview_image(z_t0, t0, grid, native, steps=lookahead)
            if review is None:
                review = review_reflection(rt, preview, prompt, cfg)["reflection"]
                outcome = str(review["outcome"])
                instruction = " ".join(str(review.get("instruction") or "").split())
                diag_action = classify_diagnosed_action(instruction) if outcome == "edit" else "none"
            _sync_device(device)
            sec_reflect = time.perf_counter() - t0_s
            nfe_preview = int(native.calls)

            _sync_device(device)
            t0_s = time.perf_counter()
            native.calls = 0
            raw_base = euler(z_t0.clone(), segment_times(grid, t0, 1.0), native)
            _sync_device(device)
            sec_base_tail = time.perf_counter() - t0_s
            nfe_base_tail = int(native.calls)

            save_image(prev_png, preview.clamp(-1.0, 1.0))
            save_image(base_png, raw_base.clamp(-1.0, 1.0))
            base_oob, base_hf = _tensor_stats(raw_base)

            arms_info = {
                "baseline": dict(
                    triggered=False,
                    image_rel=base_rel,
                    total_nfe=nfe_prefix + nfe_base_tail,
                    extra_nfe=0,
                    total_seconds=round(sec_prefix + sec_base_tail, 4),
                    oob_frac=base_oob,
                    hf_grad=base_hf,
                )
            }

        strip_rel = None
        if outcome == "edit" and instruction:
            _sync_device(device)
            t0_s = time.perf_counter()
            z_iso, clean_est, _, info_tr = isochronic_manifold_transition(
                editor,
                z_t0,
                t0,
                instruction,
                grid,
                stop_time=stop_time,
                edit_cfg_scale=edit_cfg_scale,
                anchor_prefix=null_prefix,
            )
            _sync_device(device)
            sec_tr = time.perf_counter() - t0_s
            nfe_tr = int(info_tr["anchor_nfe"] + info_tr["correction_nfe"])

            native.calls = 0
            _sync_device(device)
            t0_s = time.perf_counter()
            raw_iso = euler(z_iso, segment_times(grid, t0, 1.0), native)
            _sync_device(device)
            sec_iso_tail = time.perf_counter() - t0_s
            nfe_iso_tail = int(native.calls)

            iso_png = case_img_dir / "isochronic.png"
            save_image(iso_png, raw_iso.clamp(-1.0, 1.0))
            iso_oob, iso_hf = _tensor_stats(raw_iso)

            strip_file = strip_dir / f"{safe_id}.jpg"
            save_labeled_strip(
                [
                    (preview, f"PREVIEW({t0}, {lookahead}-step)"),
                    (raw_base, "BASELINE_G1"),
                    (clean_est, f"CLEAN_EST({stop_time})"),
                    (raw_iso, f"ISOCHRONIC({t0}->{t0})+G1"),
                ],
                strip_file,
                tile_size=320,
                quality=88,
            )
            strip_rel = str(strip_file.relative_to(out_dir))

            base_nfe_total = nfe_prefix + nfe_base_tail
            iso_total_nfe = nfe_prefix + nfe_preview + nfe_tr + nfe_iso_tail
            arms_info["isochronic"] = dict(
                triggered=True,
                image_rel=str(iso_png.relative_to(out_dir)),
                total_nfe=iso_total_nfe,
                extra_nfe=iso_total_nfe - base_nfe_total,
                total_seconds=round(sec_prefix + sec_reflect + sec_tr + sec_iso_tail, 4),
                oob_frac=iso_oob,
                hf_grad=iso_hf,
            )
        else:
            iso_total_nfe = nfe_prefix + nfe_preview + nfe_base_tail
            arms_info["isochronic"] = dict(
                triggered=False,
                image_rel=base_rel,
                total_nfe=iso_total_nfe,
                extra_nfe=nfe_preview,
                total_seconds=round(sec_prefix + sec_reflect + sec_base_tail, 4),
                oob_frac=base_oob,
                hf_grad=base_hf,
            )

        rec = dict(
            case_id=cid,
            benchmark=case["benchmark"],
            idx_in_bench=case["idx_in_bench"],
            record_id=case["record_id"],
            category=case["category"],
            prompt=prompt,
            seed=seed,
            checkpoint_step=int(ckpt_meta.get("step", -1)),
            read_time=t0,
            stop_time=stop_time,
            lookahead_steps=lookahead,
            edit_cfg_scale=edit_cfg_scale,
            outcome=outcome,
            diagnosed_action=diag_action,
            instruction=instruction,
            gated_by=review.get("gated_by"),
            gated_instruction=review.get("gated_instruction"),
            rechecked_add_subject=review.get("rechecked_add_subject"),
            description=(review.get("description") or "")[:600],
            judge_raw=(review.get("judge_raw") or "")[:400],
            strip_path=strip_rel,
            arms=arms_info,
            wall_seconds=round(time.perf_counter() - t_case0, 3),
            rank=args.rank,
        )
        with shard_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status_path.write_text(
            json.dumps({"rank": args.rank, "done": pos, "total": len(mine), "finished": False}) + "\n"
        )
        print(
            f"[gen {args.rank:02d}] ({pos}/{len(mine)}) {cid} | outcome={outcome} ({diag_action}) gated={review.get('gated_by')}",
            flush=True,
        )

    status_path.write_text(
        json.dumps({"rank": args.rank, "done": len(mine), "total": len(mine), "finished": True}) + "\n"
    )
    print(f"[gen {args.rank:02d}] FINISHED all {len(mine)} cases.", flush=True)


def parse_yesno_token(raw_text):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(raw_text or ""), flags=re.I).strip()
    m = re.search(r"\b(yes|no)\b", cleaned, flags=re.I)
    return m.group(1).lower() if m else None


def parse_geneval_pp_response(raw_text, pp_mod):
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", str(raw_text or ""), flags=re.I).strip()
    try:
        return pp_mod.extract_result(cleaned)
    except Exception:
        m = re.search(r'"correct"\s*:\s*(1|0|true|false|"1"|"0")', cleaned, flags=re.I)
        if m:
            tok = m.group(1).strip('"').lower()
            return {"correct": int(tok in ("1", "true")), "reason": cleaned[:200]}
        return None


def cmd_score_worker(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_grad_enabled(False)
    if str(args.device).startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    device = torch.device(args.device)

    gv_mod = _load_ref_module(
        "_gv_ref",
        "qwen_latent_cot/benchmarks/reference/scoring_sources/bench_tools/benchmarks/geneval/scripts/eval_geneval_qwen35.py",
    )
    pp_mod = _load_ref_module(
        "_pp_ref",
        "qwen_latent_cot/benchmarks/reference/scoring_sources/pp/eval_geneval_pp_qwen.py",
    )

    out_dir = Path(args.out).resolve()
    cases_payload = json.loads((out_dir / "t2i_cases.json").read_text())
    case_by_id = {c["case_id"]: c for c in cases_payload["cases"]}

    gen_rows = {}
    for p in sorted((out_dir / "gen_shards").glob("gen_rank*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                gen_rows[r["case_id"]] = r

    ordered_ids = [c["case_id"] for c in cases_payload["cases"] if c["case_id"] in gen_rows]
    if len(ordered_ids) != len(cases_payload["cases"]) and not args.allow_partial:
        raise RuntimeError(
            f"Generation incomplete before scoring: {len(ordered_ids)}/{len(cases_payload['cases'])}"
        )
    mine_ids = [cid for i, cid in enumerate(ordered_ids) if i % args.world == args.rank]

    score_dir = out_dir / "score_shards"
    score_dir.mkdir(parents=True, exist_ok=True)
    shard_path = score_dir / f"score_rank{args.rank:02d}.jsonl"
    status_path = score_dir / f"done_rank{args.rank:02d}.json"
    done_ids = set()
    if shard_path.exists():
        for line in shard_path.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["case_id"])

    proc = AutoProcessor.from_pretrained(args.judge_model)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.judge_model, dtype=dtype).to(device).eval()

    def _ask_vlm(pil_img, system_prompt, user_text, max_new_tokens):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def _score_single_image(case, image_path: Path):
        with Image.open(image_path) as im:
            pil_img = im.convert("RGB")
        bench = case["benchmark"]
        if bench == "geneval":
            questions = gv_mod.build_questions(case)
            if not questions:
                return {"status": "scored", "correct": 1, "answers": {}, "raw": []}
            answers = {}
            raws = []
            for q in questions:
                raw = _ask_vlm(pil_img, gv_mod.SYSTEM_PROMPT, q["question"], max_new_tokens=16)
                raws.append(raw)
                ans = parse_yesno_token(raw)
                if ans is None:
                    return {"status": "unknown", "correct": None, "answers": answers, "raw": raws}
                answers[q["id"]] = ans
            ok = int(all(answers[q["id"]] == q["must_be"] for q in questions))
            return {"status": "scored", "correct": ok, "answers": answers, "raw": raws}
        if bench == "geneval2":
            from qwen_latent_cot.benchmarks.reference import t2i_protocols as proto
            gv2_sys = proto.QA_SYSTEM + " The answer field must contain exactly yes, no, unknown, or a single numeral from 0 to 9. Do not explain."
            vqa_list = case.get("vqa_list") or []
            if not vqa_list:
                return {"status": "scored", "correct": 1, "qa_acc": 1.0, "answers": [], "raw": []}
            hits = []
            answers = []
            raws = []
            for q_text, expected_raw in vqa_list:
                raw = _ask_vlm(pil_img, gv2_sys, q_text, max_new_tokens=24)
                raws.append(raw)
                try:
                    ans = proto.interpret("geneval2", raw)
                except ValueError:
                    ans = None
                expected = proto.normalize("geneval2", str(expected_raw))
                answers.append(ans)
                if ans is None or expected is None:
                    hits.append(0)
                else:
                    hits.append(int(ans == expected))
            ok = int(all(h == 1 for h in hits))
            return {"status": "scored", "correct": ok, "qa_acc": round(sum(hits) / len(hits), 4), "answers": answers, "raw": raws}
        sys_key = "t2i_compbench_pp" if bench == "t2i_compbench_pp" else "geneval_pp"
        user_text = (
            f"Instruction:\n{case['prompt']}\n\n"
            f"Expectation checklist:\n{pp_mod.metadata_to_checklist(case['benchmark_metadata'])}"
        )
        raw = _ask_vlm(pil_img, pp_mod.SYSTEM_PROMPTS[sys_key], user_text, max_new_tokens=192)
        parsed = parse_geneval_pp_response(raw, pp_mod)
        if parsed is None:
            return {"status": "unknown", "correct": None, "raw": raw}
        return {"status": "scored", "correct": int(parsed["correct"]), "reason": parsed["reason"], "raw": raw}

    prev_score_rows = {}
    if getattr(args, "reuse_scores_from", None):
        prev_sdir = Path(args.reuse_scores_from).resolve()
        for p in sorted((prev_sdir / "score_shards").glob("score_rank*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    sr = json.loads(line)
                    prev_score_rows[sr["case_id"]] = sr

    for pos, cid in enumerate(mine_ids, 1):
        if cid in done_ids:
            continue
        t0_s = time.perf_counter()
        case = case_by_id[cid]
        grow = gen_rows[cid]

        img_cache = {}
        arm_scores = {}
        prev_sr = prev_score_rows.get(cid)
        if prev_sr is not None:
            arm_scores["baseline"] = prev_sr["scores"]["baseline"]
            if not grow["arms"]["isochronic"]["triggered"]:
                arm_scores["isochronic"] = prev_sr["scores"]["baseline"]
            else:
                arm_scores["isochronic"] = _score_single_image(case, out_dir / grow["arms"]["isochronic"]["image_rel"])
        else:
            for arm in ARMS:
                rel = grow["arms"][arm]["image_rel"]
                if rel not in img_cache:
                    img_cache[rel] = _score_single_image(case, out_dir / rel)
                arm_scores[arm] = img_cache[rel]

        rec = dict(
            case_id=cid,
            benchmark=case["benchmark"],
            record_id=case["record_id"],
            category=case["category"],
            scores=arm_scores,
            score_seconds=round(time.perf_counter() - t0_s, 3),
            rank=args.rank,
        )
        with shard_path.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        status_path.write_text(
            json.dumps({"rank": args.rank, "done": pos, "total": len(mine_ids), "finished": False}) + "\n"
        )
        print(
            f"[score {args.rank:02d}] ({pos}/{len(mine_ids)}) {cid} | "
            f"base={arm_scores['baseline']['correct']} iso={arm_scores['isochronic']['correct']}",
            flush=True,
        )

    status_path.write_text(
        json.dumps({"rank": args.rank, "done": len(mine_ids), "total": len(mine_ids), "finished": True}) + "\n"
    )
    print(f"[score {args.rank:02d}] FINISHED all {len(mine_ids)} cases.", flush=True)


def _hosts():
    hosts = [item for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item]
    if len(hosts) != 4:
        raise ValueError(f"Expected 4 hosts in VC_WORKER_HOSTS, got {len(hosts)}")
    return hosts


def _remote_cmd(host, node_idx, shell_cmd):
    if node_idx == 0:
        return ["bash", "-lc", shell_cmd]
    return [
        "ssh", "-n", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "root@" + host, "bash", "-lc", shlex.quote(shell_cmd),
    ]


def _preflight_nodes(hosts):
    for node_idx, host in enumerate(hosts):
        check = _remote_cmd(host, node_idx, "npu-smi info | grep -c 'No running processes found in NPU '")
        res = subprocess.run(check, capture_output=True, text=True, timeout=60)
        idle = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "0"
        print(f"node{node_idx} ({host}) idle NPU packages: {idle}/8", flush=True)
        if idle != "8":
            raise RuntimeError(f"Node {node_idx} ({host}) not fully idle ({idle}/8); aborting!")


def _spawn_64_workers(hosts, code_dir, helper_py, worker_py, log_dir, build_worker_args_fn, dry_run=False):
    helper_path = code_dir / "scripts" / "start_train_node.py"
    world = 4 * PER_NODE
    node_scripts = []
    for node_idx, host in enumerate(hosts):
        lines = [
            "set -euo pipefail",
            f"cd {shlex.quote(str(code_dir))}",
            f"export PYTHONPATH={shlex.quote(str(code_dir))}",
            "export PYTHONUNBUFFERED=1",
            "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "export OMP_NUM_THREADS=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        ]
        for local_rank in range(PER_NODE):
            global_rank = node_idx * PER_NODE + local_rank
            rk_log = log_dir / f"rank_{global_rank:02d}.log"
            rk_pid = log_dir / f"rank_{global_rank:02d}.pid"
            cmd = [
                helper_py,
                str(helper_path),
                "--log", str(rk_log),
                "--pid-file", str(rk_pid),
                "--",
                worker_py,
                *build_worker_args_fn(local_rank, global_rank, world),
            ]
            lines.append(shlex.join(cmd))
            if local_rank + 1 < PER_NODE:
                lines.append("sleep 0.25")
        lines.append("echo NODE_LAUNCHED_16")
        node_scripts.append((node_idx, host, "\n".join(lines)))

    if dry_run:
        for node_idx, host, shell in node_scripts:
            print(f"=== NODE {node_idx} ({host}) ===\n{shell}\n")
        return

    for node_idx, host, shell in node_scripts:
        res = subprocess.run(_remote_cmd(host, node_idx, shell), capture_output=True, text=True, timeout=60)
        if res.returncode != 0 or "NODE_LAUNCHED_16" not in res.stdout:
            raise RuntimeError(f"Failed launching node {node_idx}: {res.stdout}\n{res.stderr}")


def cmd_launch_gen64(args):
    hosts = _hosts()
    out_dir = Path(args.out).resolve()
    if out_dir.exists() and not args.resume:
        raise ValueError(f"Output directory exists (pass --resume): {out_dir}")
    _preflight_nodes(hosts)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "gen_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    cases_file = out_dir / "t2i_cases.json"
    if not cases_file.exists():
        payload = build_t2i_cases(
            args.geneval if not args.skip_geneval else None,
            args.geneval_pp if not args.skip_geneval_pp else None,
            args.geneval2 if getattr(args, "geneval2", None) else None,
            args.t2i_compbench if getattr(args, "t2i_compbench", None) else None,
        )
        cases_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Prepared {payload['total_cases']} cases -> {cases_file}", flush=True)

    code_dir = Path(args.code).resolve()
    script_path = code_dir / "scripts" / "bench_t2i_closed_loop64.py"

    def _worker_args(lr, gr, w):
        w_args = [
            str(script_path), "gen-worker",
            "--model", args.model,
            "--source", args.source,
            "--config", str(Path(args.config).resolve()),
            "--checkpoint", str(Path(args.checkpoint).resolve()),
            "--out", str(out_dir),
            "--device", f"npu:{lr}",
            "--rank", str(gr),
            "--world", str(w),
        ]
        if args.stop_time is not None:
            w_args += ["--stop-time", str(args.stop_time)]
        if args.lookahead_steps is not None:
            w_args += ["--lookahead-steps", str(args.lookahead_steps)]
        if args.edit_cfg_scale is not None:
            w_args += ["--edit-cfg-scale", str(args.edit_cfg_scale)]
        if args.no_recheck_add:
            w_args.append("--no-recheck-add")
        if args.no_consistency_gate:
            w_args.append("--no-consistency-gate")
        if getattr(args, "reflection_cache", None):
            w_args += ["--reflection-cache", str(Path(args.reflection_cache).resolve())]
        if getattr(args, "reuse_baseline_from", None):
            w_args += ["--reuse-baseline-from", str(Path(args.reuse_baseline_from).resolve())]
        return w_args

    _spawn_64_workers(hosts, code_dir, args.python, args.python, log_dir, _worker_args, dry_run=args.dry_run)
    print(f"Launched 64 T2I closed-loop generation workers -> {out_dir}", flush=True)


def cmd_launch_score64(args):
    hosts = _hosts()
    out_dir = Path(args.out).resolve()
    _preflight_nodes(hosts)
    log_dir = out_dir / "score_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    code_dir = Path(args.code).resolve()
    script_path = code_dir / "scripts" / "bench_t2i_closed_loop64.py"
    def _score_worker_args(lr, gr, w):
        s_args = [
            str(script_path), "score-worker",
            "--judge-model", args.judge_model,
            "--out", str(out_dir),
            "--device", f"npu:{lr}",
            "--rank", str(gr),
            "--world", str(w),
        ]
        if getattr(args, "reuse_scores_from", None):
            s_args += ["--reuse-scores-from", str(Path(args.reuse_scores_from).resolve())]
        return s_args

    _spawn_64_workers(
        hosts,
        code_dir,
        args.helper_python,
        args.judge_python,
        log_dir,
        _score_worker_args,
        dry_run=args.dry_run,
    )
    print(f"Launched 64 independent Qwen3-VL scoring workers -> {out_dir}", flush=True)


def _summarize_benchmark(joined_rows, bench_name):
    from _time_probe_common import mean_or_none, paired_bootstrap_ci

    sub = [r for r in joined_rows if r["benchmark"] == bench_name]
    if not sub:
        return {}

    valid_ids = [
        r["case_id"]
        for r in sub
        if all(r["scores"][a].get("status") == "scored" and r["scores"][a].get("correct") is not None for a in ARMS)
    ]
    valid_rows = [r for r in sub if r["case_id"] in set(valid_ids)]

    cats = list(GENEVAL_CATS) if bench_name == "geneval" else sorted({r["category"] for r in valid_rows})
    trig_rows = [r for r in valid_rows if r["outcome"] == "edit"]
    base_fails = [r for r in valid_rows if int(r["scores"]["baseline"]["correct"]) == 0]
    base_passes = [r for r in valid_rows if int(r["scores"]["baseline"]["correct"]) == 1]
    trig_base_fails = [r for r in trig_rows if int(r["scores"]["baseline"]["correct"]) == 0]
    trig_base_passes = [r for r in trig_rows if int(r["scores"]["baseline"]["correct"]) == 1]

    arm_summaries = {}
    for arm in ARMS:
        cat_acc = {}
        for c in cats:
            c_vals = [float(r["scores"][arm]["correct"]) for r in valid_rows if r["category"] == c]
            cat_acc[c] = round(sum(c_vals) / len(c_vals), 4) if c_vals else None
        present = [v for v in cat_acc.values() if v is not None]
        macro = sum(present) / len(present) if present else 0.0
        micro_vals = [float(r["scores"][arm]["correct"]) for r in valid_rows]
        micro_m, micro_lo, micro_hi, _ = paired_bootstrap_ci(micro_vals, n_boot=2000, seed=17)

        fixed = sum(1 for r in trig_base_fails if int(r["scores"][arm]["correct"]) == 1)
        damaged = sum(1 for r in trig_base_passes if int(r["scores"][arm]["correct"]) == 0)

        arm_summaries[arm] = dict(
            scored_count=len(valid_rows),
            macro_accuracy=round(macro, 4),
            micro_accuracy=round(micro_m, 4) if micro_m is not None else None,
            micro_ci95=[round(micro_lo, 4), round(micro_hi, 4)] if micro_m is not None else None,
            category_accuracy=cat_acc,
            fixed_on_trig_fail=fixed,
            trig_base_fails=len(trig_base_fails),
            fix_rate_on_trig_fail=round(fixed / len(trig_base_fails), 4) if trig_base_fails else None,
            damaged_on_trig_pass=damaged,
            trig_base_passes=len(trig_base_passes),
            damage_rate_on_trig_pass=round(damaged / len(trig_base_passes), 4) if trig_base_passes else None,
            net_fixed_cases=fixed - damaged,
            mean_total_nfe=round(mean_or_none([r["arms"][arm]["total_nfe"] for r in valid_rows]), 2),
            mean_extra_nfe=round(mean_or_none([r["arms"][arm]["extra_nfe"] for r in valid_rows]), 2),
            mean_seconds=round(mean_or_none([r["arms"][arm]["total_seconds"] for r in valid_rows]), 3),
            mean_hf_grad=round(mean_or_none([r["arms"][arm]["hf_grad"] for r in valid_rows]), 5),
            mean_oob_frac=round(mean_or_none([r["arms"][arm]["oob_frac"] for r in valid_rows]), 5),
            triggered_hf_grad=round(mean_or_none([r["arms"][arm]["hf_grad"] for r in trig_rows]), 5) if trig_rows else None,
        )

    def _delta(subset=None):
        rows_use = valid_rows if subset is None else subset
        diffs = [float(r["scores"]["isochronic"]["correct"]) - float(r["scores"]["baseline"]["correct"]) for r in rows_use]
        m, lo, hi, n = paired_bootstrap_ci(diffs, n_boot=2000, seed=31)
        return None if m is None else {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": n}

    outcomes = Counter(r["outcome"] for r in valid_rows)
    gates = Counter(r.get("gated_by") for r in valid_rows if r.get("gated_by"))
    by_action = {}
    for act in sorted({r["diagnosed_action"] for r in trig_rows}):
        arows = [r for r in trig_rows if r["diagnosed_action"] == act]
        by_action[act] = {
            "count": len(arows),
            "baseline_acc": round(sum(float(r["scores"]["baseline"]["correct"]) for r in arows) / len(arows), 4),
            "isochronic_acc": round(sum(float(r["scores"]["isochronic"]["correct"]) for r in arows) / len(arows), 4),
        }

    return dict(
        total_manifest=len(sub),
        common_valid_scored=len(valid_rows),
        outcomes=dict(outcomes),
        gated_counts=dict(gates),
        rechecked_add_cases=sum(1 for r in valid_rows if r.get("rechecked_add_subject")),
        trigger_rate=round(len(trig_rows) / len(valid_rows), 4) if valid_rows else 0.0,
        parse_fail_rate=round(sum(1 for r in sub if r["outcome"] in ("failed_parse", "failed_unconverted")) / len(sub), 4),
        error_catch_recall=round(len(trig_base_fails) / len(base_fails), 4) if base_fails else None,
        false_trigger_rate=round(len(trig_base_passes) / len(base_passes), 4) if base_passes else None,
        arms=arm_summaries,
        by_diagnosed_action=by_action,
        paired_delta_all=_delta(),
        paired_delta_triggered=_delta(trig_rows),
    )


def _write_html_gallery(joined_rows, summary, html_path):
    trig_items = []
    for r in joined_rows:
        if not r.get("strip_path"):
            continue
        b = int(r["scores"]["baseline"].get("correct") or 0)
        iso = int(r["scores"]["isochronic"].get("correct") or 0)
        trig_items.append(
            dict(
                cid=r["case_id"],
                bench=r["benchmark"],
                cat=r["category"],
                act=r["diagnosed_action"],
                prompt=r["prompt"],
                ins=r["instruction"],
                strip=r["strip_path"],
                b=b,
                iso=iso,
            )
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Isochronic Closed-Loop T2I Gallery</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#141418; color:#ececf1; margin:20px; }}
.card {{ background:#1f2028; border:1px solid #323442; border-radius:8px; padding:12px; margin-bottom:16px; }}
.card img {{ max-width:100%; border-radius:4px; margin-top:8px; }}
.meta {{ font-size:13px; color:#b8bac8; }}
pre {{ background:#181920; padding:12px; border-radius:6px; overflow-x:auto; font-size:12px; }}
</style></head><body>
<h2>Strict Isochronic Manifold Closed-Loop Gallery ({len(trig_items)} Triggered Cases)</h2>
<pre>{json.dumps({k: summary.get(k) for k in ("integrity", "geneval", "geneval_pp")}, indent=2)}</pre>
<div id="list"></div>
<script>
const ITEMS = {json.dumps(trig_items)};
document.getElementById('list').innerHTML = ITEMS.map(x => `
  <div class="card">
    <div class="meta"><b>${{x.cid}}</b> | ${{x.bench}} / ${{x.cat}} | action=<b>${{x.act}}</b> | BASE=${{x.b}} -> ISO=${{x.iso}}</div>
    <div class="meta"><b>Prompt:</b> ${{x.prompt}}</div>
    <div class="meta"><b>UND Edit:</b> ${{x.ins}}</div>
    <img loading="lazy" src="${{x.strip}}"/>
  </div>`).join('');
</script></body></html>"""
    Path(html_path).write_text(html)


def cmd_merge(args):
    out_dir = Path(args.out).resolve()
    cases_payload = json.loads((out_dir / "t2i_cases.json").read_text())
    expected = [c["case_id"] for c in cases_payload["cases"]]

    gen_map = {}
    for p in sorted((out_dir / "gen_shards").glob("gen_rank*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                gen_map[r["case_id"]] = r

    score_map = {}
    for p in sorted((out_dir / "score_shards").glob("score_rank*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                score_map[r["case_id"]] = r

    missing_gen = [cid for cid in expected if cid not in gen_map]
    missing_score = [cid for cid in expected if cid not in score_map]
    integrity = dict(
        expected=len(expected),
        generated=len(gen_map),
        scored=len(score_map),
        missing_gen=len(missing_gen),
        missing_score=len(missing_score),
        complete=(len(missing_gen) == 0 and len(missing_score) == 0),
    )
    (out_dir / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    if not integrity["complete"] and not args.allow_partial:
        raise RuntimeError(f"Incomplete benchmark run: {integrity}")

    joined = [
        {**gen_map[c["case_id"]], "scores": score_map[c["case_id"]]["scores"]}
        for c in cases_payload["cases"]
        if c["case_id"] in gen_map and c["case_id"] in score_map
    ]

    summary = {
        "protocol": cases_payload["protocol"],
        "judge": "Qwen3-VL-8B-Instruct (independent VLM, official clause/checklist rubrics; non-Mask2Former/GPT4.1)",
        "integrity": integrity,
        "geneval": _summarize_benchmark(joined, "geneval"),
        "geneval_pp": _summarize_benchmark(joined, "geneval_pp"),
        "geneval2": _summarize_benchmark(joined, "geneval2"),
        "t2i_compbench_pp": _summarize_benchmark(joined, "t2i_compbench_pp"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_html_gallery(joined, summary, out_dir / "index.html")

    lines = [
        "# Strict Isochronic Manifold Closed-Loop 64-NPU Benchmark Summary",
        "",
        f"- Integrity: `gen={integrity['generated']}/{integrity['expected']}`, `score={integrity['scored']}/{integrity['expected']}` (`complete={integrity['complete']}`)",
        "- Independent Judge: `Qwen3-VL-8B-Instruct`",
        "",
    ]
    for bkey, title in (
        ("geneval", "GenEval Full (553 Prompts)"),
        ("geneval_pp", "GenEval++ Full (280 Prompts)"),
        ("geneval2", "GenEval2 Full (800 Prompts)"),
        ("t2i_compbench_pp", "T2I-CompBench++ Full (2400 Prompts)"),
    ):
        bs = summary.get(bkey) or {}
        if not bs:
            continue
        lines += [
            f"## {title}",
            f"- Trigger rate: `{bs['trigger_rate']}` | Parse fail: `{bs['parse_fail_rate']}` | Error-catch recall: `{bs['error_catch_recall']}` | False-trigger rate: `{bs['false_trigger_rate']}` | Gated: `{bs['gated_counts']}`",
            "",
            "| Arm | Macro Acc ↑ | Micro Acc ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Triggered HF | Mean NFE | Mean Sec |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for arm in ARMS:
            a = bs["arms"][arm]
            lines.append(
                f"| `{arm}` | **{a['macro_accuracy']}** | {a['micro_accuracy']} | "
                f"{a['fixed_on_trig_fail']}/{a['trig_base_fails']} ({a['fix_rate_on_trig_fail']}) | "
                f"{a['damaged_on_trig_pass']}/{a['trig_base_passes']} ({a['damage_rate_on_trig_pass']}) | "
                f"{a['net_fixed_cases']:+d} | {a['triggered_hf_grad']} | {a['mean_total_nfe']} | {a['mean_seconds']} |"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def _add_override_args(p):
    p.add_argument("--stop-time", type=float, default=None)
    p.add_argument("--lookahead-steps", type=int, default=None)
    p.add_argument("--edit-cfg-scale", type=float, default=None)
    p.add_argument("--no-recheck-add", action="store_true")
    p.add_argument("--no-consistency-gate", action="store_true")
    p.add_argument("--reflection-cache", default=None)
    p.add_argument("--reuse-baseline-from", default=None)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--geneval", default=DEFAULT_GENEVAL)
    p_prep.add_argument("--geneval-pp", default=DEFAULT_GENEVAL_PP)
    p_prep.add_argument("--geneval2", default=None)
    p_prep.add_argument("--t2i-compbench", default=None)
    p_prep.add_argument("--skip-geneval", action="store_true")
    p_prep.add_argument("--skip-geneval-pp", action="store_true")
    p_prep.add_argument("--out", required=True)

    p_gw = sub.add_parser("gen-worker")
    p_gw.add_argument("--model", default=DEFAULT_MODEL)
    p_gw.add_argument("--source", default=DEFAULT_SOURCE)
    p_gw.add_argument("--config", default=DEFAULT_CONFIG)
    p_gw.add_argument("--checkpoint", required=True)
    p_gw.add_argument("--out", required=True)
    _add_override_args(p_gw)
    p_gw.add_argument("--device", default="npu:0")
    p_gw.add_argument("--rank", type=int, required=True)
    p_gw.add_argument("--world", type=int, required=True)

    p_sw = sub.add_parser("score-worker")
    p_sw.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p_sw.add_argument("--out", required=True)
    p_sw.add_argument("--device", default="npu:0")
    p_sw.add_argument("--rank", type=int, required=True)
    p_sw.add_argument("--world", type=int, required=True)
    p_sw.add_argument("--allow-partial", action="store_true")
    p_sw.add_argument("--reuse-scores-from", default=None)

    p_lg = sub.add_parser("launch-gen64")
    p_lg.add_argument("--geneval", default=DEFAULT_GENEVAL)
    p_lg.add_argument("--geneval-pp", default=DEFAULT_GENEVAL_PP)
    p_lg.add_argument("--geneval2", default=None)
    p_lg.add_argument("--t2i-compbench", default=None)
    p_lg.add_argument("--skip-geneval", action="store_true")
    p_lg.add_argument("--skip-geneval-pp", action="store_true")
    p_lg.add_argument("--model", default=DEFAULT_MODEL)
    p_lg.add_argument("--source", default=DEFAULT_SOURCE)
    p_lg.add_argument("--config", default=DEFAULT_CONFIG)
    p_lg.add_argument("--checkpoint", required=True)
    p_lg.add_argument("--out", required=True)
    _add_override_args(p_lg)
    p_lg.add_argument("--code", default=str(ROOT))
    p_lg.add_argument("--python", default=DEFAULT_U15_PY)
    p_lg.add_argument("--resume", action="store_true")
    p_lg.add_argument("--dry-run", action="store_true")

    p_ls = sub.add_parser("launch-score64")
    p_ls.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p_ls.add_argument("--out", required=True)
    p_ls.add_argument("--code", default=str(ROOT))
    p_ls.add_argument("--helper-python", default=DEFAULT_U15_PY)
    p_ls.add_argument("--judge-python", default=DEFAULT_QWEN_PY)
    p_ls.add_argument("--reuse-scores-from", default=None)
    p_ls.add_argument("--dry-run", action="store_true")

    p_mg = sub.add_parser("merge")
    p_mg.add_argument("--out", required=True)
    p_mg.add_argument("--allow-partial", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    {
        "prepare": cmd_prepare,
        "gen-worker": cmd_gen_worker,
        "score-worker": cmd_score_worker,
        "launch-gen64": cmd_launch_gen64,
        "launch-score64": cmd_launch_score64,
        "merge": cmd_merge,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
