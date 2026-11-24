"""64-NPU End-to-End Closed-Loop Causal Benchmark (Arms A/B/C/E) for reflective_t2i_v1.

Decoupled 4-phase execution on 4x16 Ascend 910C NPUs (zero VRAM contention between
Generator and external VLM Observer/Judge):
  1. prepare         : Freeze GenEval2 (800) + GenEval++ (280) prompts + shuffled causal pairs.
  2. gen-prefix      : Generate shared native prefix (0 -> p0), 4-step preview, and Arm A (A_native).
  3. reflect         : Run decoupled external VLM Observer + ReflectionPolicy (single-pass JSON
                       + bidirectional add/remove focused recheck + confidence/arithmetic gates).
  4. gen-arms        : On triggered prompts, generate Arm B (VLM Rewrite+Resample),
                       Arm C (IsochronicTransitionV2 Ours), and Arm E (Shuffled Feedback).
  5. score           : Score all 4 arms with independent VLM Judge (exact GenEval2 / GenEval++ rubrics).
  6. merge           : Compute Exact Acc, Atomic QA Acc, Fix/Damage/Net Fixed, paired bootstrap CIs,
                       per-action breakdown (add/remove/attr/move), HF Grad, and summary.md/json.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("REFLECTIVE_WORKSPACE", ROOT.parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_latent_cot.reflective_t2i.core import (
    IsochronicTransitionV2,
    LatentState,
    ReflectionDecision,
    ReflectionPolicy,
    ReflectionPolicyConfig,
    TransitionConfig,
    compile_concise_imperative,
)
from qwen_latent_cot.reflective_t2i.evaluation import compute_hf_grad
from qwen_latent_cot.reflective_t2i.generators.diffusers import build_generator
from qwen_latent_cot.reflective_t2i.observers import LocalHFObserver

DEFAULT_U15_PY = str(WORKSPACE / "venvs/u15-NPU/bin/python")
DEFAULT_SCORE_PY = str(WORKSPACE / "envs/qwen3vl-score/bin/python")
DEFAULT_VLM_MODEL = "/data/zyd_workspace/models/Qwen3-VL-8B-Instruct"
DEFAULT_GENEVAL_PP = (
    "/data/zyd_workspace/isochronic_data_20261107/benchmark_assets/frozen/outputs/"
    "0_eval62326_qwen27b_sixbench_20261019/foundation_v1/inputs/score_manifests/geneval_pp.jsonl"
)
DEFAULT_GENEVAL2 = str(WORKSPACE / "code/anchor-search-20261104/data/geneval2_800.jsonl")
DEFAULT_ZIMAGE_CORE = "/data/zyd_workspace/zimage-isochronic-correction/src"
DEFAULT_SD35_DEPS = str(WORKSPACE / "venvs/sd35-deps")

ARMS = ("A_native", "B_rewrite_resample", "C_reflective", "E_shuffled_feedback")
PER_NODE = 16


def _setup_device(device: str) -> str:
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    return device


def _sync(device: str) -> None:
    if device.startswith("npu") and hasattr(torch, "npu"):
        torch.npu.synchronize(device)
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _pil_hf_grad(pil_img: Image.Image) -> float:
    import numpy as np

    arr = torch.from_numpy(np.asarray(pil_img.convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 127.5 - 1.0
    return round(compute_hf_grad(arr), 5)


def _build_generator(cfg, device, *, attach_corrector):
    return build_generator(cfg, device, attach_corrector=attach_corrector, default_rank=32,
                           default_scope="all", default_sd_size=512, default_zimage_src=DEFAULT_ZIMAGE_CORE)


def cmd_prepare(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    limit = int(getattr(args, "limit_per_bench", 0) or 0)
    items = []
    if args.geneval2 and not args.skip_geneval2:
        gv2_count = 0
        for idx, line in enumerate(Path(args.geneval2).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = str(r.get("prompt") or r.get("raw_prompt") or r.get("instruction") or "").strip()
            rid = str(r.get("prompt_id") or r.get("record_id") or f"geneval2_{idx:04d}")
            cat = str(r.get("category") or (f"atoms_{r['bucket']}" if r.get("bucket") is not None else "all"))
            items.append(
                {
                    "case_id": f"geneval2:{rid}",
                    "benchmark": "geneval2",
                    "idx_in_bench": idx,
                    "record_id": rid,
                    "category": cat,
                    "prompt": prompt,
                    "seed": int(r.get("seed", 20260900 + idx)),
                    "vqa_list": r["vqa_list"],
                }
            )
            gv2_count += 1
            if limit > 0 and gv2_count >= limit:
                break
    if args.geneval_pp and not args.skip_geneval_pp:
        pp_count = 0
        for idx, line in enumerate(Path(args.geneval_pp).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = str(r.get("required_prompt") or r.get("instruction") or "").strip()
            cat = str(r.get("category") or r.get("benchmark_metadata", {}).get("tag") or "other")
            items.append(
                {
                    "case_id": f"geneval_pp:{r['record_id']}",
                    "benchmark": "geneval_pp",
                    "idx_in_bench": idx,
                    "record_id": str(r["record_id"]),
                    "category": cat,
                    "prompt": prompt,
                    "seed": int(r["seed"]),
                    "benchmark_metadata": r["benchmark_metadata"],
                }
            )
            pp_count += 1
            if limit > 0 and pp_count >= limit:
                break

    payload = {
        "protocol": "reflective_t2i_v1_bench64",
        "benchmarks": dict(Counter(x["benchmark"] for x in items)),
        "total_cases": len(items),
        "cases": items,
    }
    (out_dir / "cases.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared {len(items)} cases ({payload['benchmarks']}) -> {out_dir / 'cases.json'}", flush=True)


def cmd_gen_prefix(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = _setup_device(args.device)
    adapter = _build_generator(cfg, device, attach_corrector=False)

    out_dir = Path(args.out).resolve()
    cases = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    mine = [c for i, c in enumerate(cases) if i % int(args.world) == int(args.rank)]

    gcfg, rcfg = cfg["generator"], cfg["reflection"]
    p0 = float(rcfg["read_progress"])
    gscale = float(gcfg.get("guidance_scale", 4.0))
    cfg_int = tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80)))
    lookahead = int(rcfg.get("lookahead_steps", 4))
    null_cond = adapter.encode_prompt(None)

    shard_dir = out_dir / "prefix_shards"
    img_root = out_dir / "images"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"prefix_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    with torch.no_grad():
        for pos, case in enumerate(mine, 1):
            cid = case["case_id"]
            if cid in done:
                continue
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", cid)
            cdir = img_root / case["benchmark"] / safe
            cdir.mkdir(parents=True, exist_ok=True)

            _sync(device)
            t0 = time.perf_counter()
            cond = adapter.encode_prompt(case["prompt"])
            z0 = adapter.sample_initial_noise(seed=int(case["seed"]))
            state_p0, _ = adapter.sample_until(
                z0, p0, cond, null_cond, gscale, cfg_interval=cfg_int, return_clean_at_end=False
            )
            _sync(device)
            sec_prefix = time.perf_counter() - t0

            t1 = time.perf_counter()
            prev_img = adapter.lookahead_preview(
                state_p0, cond, null_cond, gscale, steps=lookahead, cfg_interval=cfg_int
            )
            _sync(device)
            sec_prev = time.perf_counter() - t1

            t2 = time.perf_counter()
            nat_lat = adapter.continue_native(state_p0, cond, null_cond, gscale, cfg_interval=cfg_int)
            nat_img = adapter.decode_preview(nat_lat)
            _sync(device)
            sec_tail = time.perf_counter() - t2

            prev_path = cdir / "preview.png"
            nat_path = cdir / "A_native.png"
            prev_img.save(prev_path)
            nat_img.save(nat_path)

            rec = {
                "case_id": cid,
                "benchmark": case["benchmark"],
                "preview_rel": str(prev_path.relative_to(out_dir)),
                "native_rel": str(nat_path.relative_to(out_dir)),
                "native_hf": _pil_hf_grad(nat_img),
                "sec_prefix": round(sec_prefix, 4),
                "sec_preview": round(sec_prev, 4),
                "sec_native_tail": round(sec_tail, 4),
            }
            with shard_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pos == 1 or pos % 5 == 0 or pos == len(mine):
                print(f"[prefix r{args.rank:02d}] ({pos}/{len(mine)}) {cid} hf={rec['native_hf']} {sec_prefix+sec_tail:.1f}s", flush=True)


def cmd_reflect(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rcfg = cfg["reflection"]
    out_dir = Path(args.out).resolve()
    cases = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    prefix_rows = {}
    for p in sorted((out_dir / "prefix_shards").glob("prefix_rank*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                prefix_rows[r["case_id"]] = r

    mine = [c for i, c in enumerate(cases) if i % int(args.world) == int(args.rank)]
    obs = LocalHFObserver(args.observer_model, device=args.device)
    policy = ReflectionPolicy(
        ReflectionPolicyConfig(
            focused_recheck_actions=tuple(rcfg.get("focused_recheck_actions", ("add", "remove"))),
            confidence_threshold=float(rcfg.get("confidence_threshold", 0.75)),
            contradiction_gate=bool(rcfg.get("contradiction_gate", True)),
        )
    )

    rdir = out_dir / "reflect_shards"
    rdir.mkdir(parents=True, exist_ok=True)
    shard_path = rdir / f"reflect_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    for pos, case in enumerate(mine, 1):
        cid = case["case_id"]
        if cid in done:
            continue
        t0 = time.perf_counter()
        prev_path = out_dir / prefix_rows[cid]["preview_rel"]
        with Image.open(prev_path) as im:
            dec = policy.reflect(obs, im.convert("RGB"), case["prompt"])
        sec = round(time.perf_counter() - t0, 4)
        rec = {
            "case_id": cid,
            "benchmark": case["benchmark"],
            "outcome": dec.outcome,
            "action": dec.action,
            "instruction": dec.instruction,
            "target_rewrite": dec.target_rewrite,
            "coarse_region_hint": dec.coarse_region_hint,
            "confidence": dec.confidence,
            "gated_by": dec.gated_by,
            "gated_instruction": dec.gated_instruction,
            "sec_reflect": sec,
        }
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if pos == 1 or pos % 5 == 0 or pos == len(mine):
            print(f"[reflect r{args.rank:02d}] ({pos}/{len(mine)}) {cid} -> {dec.outcome} ({dec.action}) gated={dec.gated_by} {sec:.1f}s", flush=True)


def cmd_gen_arms(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = _setup_device(args.device)
    adapter = _build_generator(cfg, device, attach_corrector=True)
    sd = torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True)
    adapter.corrector.load_state_dict_trainable(sd, strict=True)

    out_dir = Path(args.out).resolve()
    cases = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    prefix_rows = {
        r["case_id"]: r
        for p in sorted((out_dir / "prefix_shards").glob("prefix_rank*.jsonl"))
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }
    ref_rows = {
        r["case_id"]: r
        for p in sorted((out_dir / "reflect_shards").glob("reflect_rank*.jsonl"))
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }

    # Build deterministic derangement of triggered instructions for Arm E (Shuffled Feedback)
    trig_ids = sorted(cid for cid, r in ref_rows.items() if r["outcome"] == "edit" and r["instruction"])
    shuffled_map = {}
    if len(trig_ids) >= 2:
        for i, cid in enumerate(trig_ids):
            partner = trig_ids[(i + len(trig_ids) // 2 + 1) % len(trig_ids)]
            shuffled_map[cid] = ref_rows[partner]["instruction"]

    gcfg, rcfg, ccfg = cfg["generator"], cfg["reflection"], cfg["corrector"]
    p0 = float(rcfg["read_progress"])
    p_stop = float(rcfg["stop_progress"])
    gscale = float(gcfg.get("guidance_scale", 4.0))
    cfg_int = tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80)))
    cond_mode = ccfg.get("condition_mode", "feedback_only")
    transition = IsochronicTransitionV2(
        TransitionConfig(
            read_progress=p0,
            stop_progress=p_stop,
            anchor_mode=ccfg.get("anchor_mode", "conditional_native"),
            edit_cfg_scale=float(ccfg.get("edit_cfg_scale", 1.4)),
            trust_region_rho=float(ccfg.get("trust_region_rho", 0.65)),
            noise_refresh=float(ccfg.get("noise_refresh", 0.0)),
        )
    )
    null_cond = adapter.encode_prompt(None)

    adir = out_dir / "arms_shards"
    adir.mkdir(parents=True, exist_ok=True)
    shard_path = adir / f"arms_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    mine = [c for i, c in enumerate(cases) if i % int(args.world) == int(args.rank)]
    with torch.no_grad():
        for pos, case in enumerate(mine, 1):
            cid = case["case_id"]
            if cid in done:
                continue
            pr = prefix_rows[cid]
            rr = ref_rows[cid]
            nat_rel = pr["native_rel"]
            triggered = rr["outcome"] == "edit" and bool(rr["instruction"])

            arms_meta = {
                "A_native": {
                    "image_rel": nat_rel,
                    "triggered": False,
                    "hf_grad": pr["native_hf"],
                    "seconds": round(pr["sec_prefix"] + pr["sec_native_tail"], 4),
                }
            }

            if not triggered:
                for a in ("B_rewrite_resample", "C_reflective", "E_shuffled_feedback"):
                    arms_meta[a] = {
                        "image_rel": nat_rel,
                        "triggered": False,
                        "hf_grad": pr["native_hf"],
                        "seconds": round(pr["sec_prefix"] + pr["sec_preview"] + rr["sec_reflect"] + pr["sec_native_tail"], 4),
                    }
            else:
                cdir = (out_dir / nat_rel).parent
                seed = int(case["seed"])
                prompt = case["prompt"]
                orig_cond = adapter.encode_prompt(prompt)
                z0 = adapter.sample_initial_noise(seed=seed)

                # Shared prefix 0 -> p0 + cached conditional clean estimate
                state_p0, cached_clean = adapter.sample_until(
                    z0, p0, orig_cond, null_cond, gscale, cfg_interval=cfg_int, return_clean_at_end=True
                )

                # Arm B: VLM Prompt Rewrite + Resample (same seed/noise)
                _sync(device)
                tb = time.perf_counter()
                rw_prompt = rr.get("target_rewrite") or prompt
                rw_cond = adapter.encode_prompt(rw_prompt)
                st_b, _ = adapter.sample_until(
                    z0.clone(), p0, rw_cond, null_cond, gscale, cfg_interval=cfg_int, return_clean_at_end=False
                )
                lat_b = adapter.continue_native(st_b, rw_cond, null_cond, gscale, cfg_interval=cfg_int)
                img_b = adapter.decode_preview(lat_b)
                _sync(device)
                sec_b = time.perf_counter() - tb
                path_b = cdir / "B_rewrite_resample.png"
                img_b.save(path_b)
                arms_meta["B_rewrite_resample"] = {
                    "image_rel": str(path_b.relative_to(out_dir)),
                    "triggered": True,
                    "hf_grad": _pil_hf_grad(img_b),
                    "seconds": round(pr["sec_prefix"] + pr["sec_preview"] + rr["sec_reflect"] + sec_b, 4),
                }

                # Arm C: reflective_t2i_v1 IsochronicTransitionV2 (Ours)
                _sync(device)
                tc = time.perf_counter()
                inst_c = rr["instruction"]
                hint = rr.get("coarse_region_hint") or ""
                if hint and hint.lower() not in inst_c.lower():
                    inst_c = f"{inst_c.rstrip('.')} ({hint})."
                edit_cond_c = adapter.encode_edit_condition(
                    feedback=inst_c,
                    original_prompt=prompt,
                    target_rewrite=rw_prompt,
                    condition_mode=cond_mode,
                )
                tr_c = transition.apply(
                    adapter,
                    state_p0,
                    edit_cond_c,
                    orig_cond,
                    null_cond,
                    cached_native_clean=cached_clean,
                    generator=torch.Generator(device=device).manual_seed(seed * 97),
                )
                lat_c = adapter.continue_native(tr_c.state, orig_cond, null_cond, gscale, cfg_interval=cfg_int)
                img_c = adapter.decode_preview(lat_c)
                _sync(device)
                sec_c = time.perf_counter() - tc
                path_c = cdir / "C_reflective.png"
                img_c.save(path_c)
                arms_meta["C_reflective"] = {
                    "image_rel": str(path_c.relative_to(out_dir)),
                    "triggered": True,
                    "hf_grad": _pil_hf_grad(img_c),
                    "edit_clean_rms": tr_c.telemetry["edit_clean_rms"],
                    "seconds": round(pr["sec_prefix"] + pr["sec_preview"] + rr["sec_reflect"] + sec_c, 4),
                }

                # Arm E: Shuffled Feedback Causal Control (never leak true original_prompt into edit_cond_e)
                _sync(device)
                te = time.perf_counter()
                shuf_inst = shuffled_map.get(cid, "Change the lighting to sunset.")
                edit_cond_e = adapter.encode_edit_condition(
                    feedback=shuf_inst,
                    original_prompt=None,
                    target_rewrite=None,
                    condition_mode="feedback_only",
                )
                tr_e = transition.apply(
                    adapter,
                    state_p0,
                    edit_cond_e,
                    orig_cond,
                    null_cond,
                    cached_native_clean=cached_clean,
                    generator=torch.Generator(device=device).manual_seed(seed * 97),
                )
                lat_e = adapter.continue_native(tr_e.state, orig_cond, null_cond, gscale, cfg_interval=cfg_int)
                img_e = adapter.decode_preview(lat_e)
                _sync(device)
                sec_e = time.perf_counter() - te
                path_e = cdir / "E_shuffled_feedback.png"
                img_e.save(path_e)
                arms_meta["E_shuffled_feedback"] = {
                    "image_rel": str(path_e.relative_to(out_dir)),
                    "triggered": True,
                    "hf_grad": _pil_hf_grad(img_e),
                    "shuffled_instruction": shuf_inst,
                    "seconds": round(pr["sec_prefix"] + pr["sec_preview"] + rr["sec_reflect"] + sec_e, 4),
                }

            rec = {
                "case_id": cid,
                "benchmark": case["benchmark"],
                "category": case["category"],
                "outcome": rr["outcome"],
                "action": rr["action"],
                "gated_by": rr["gated_by"],
                "instruction": rr["instruction"],
                "arms": arms_meta,
            }
            with shard_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pos == 1 or pos % 5 == 0 or pos == len(mine):
                print(f"[arms r{args.rank:02d}] ({pos}/{len(mine)}) {cid} trig={triggered} ({rr['action']})", flush=True)


def _load_pp_module():
    common_dir = ROOT / "qwen_latent_cot/benchmarks/reference/scoring_sources/bench_tools/common"
    if str(common_dir) not in sys.path:
        sys.path.insert(0, str(common_dir))
    spec = importlib.util.spec_from_file_location(
        "_pp_ref", ROOT / "qwen_latent_cot/benchmarks/reference/scoring_sources/pp/eval_geneval_pp_qwen.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_judge(judge_model, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from qwen_latent_cot.benchmarks.reference import t2i_protocols as proto

    dev = torch.device(device)
    pp_mod = _load_pp_module()

    proc = AutoProcessor.from_pretrained(judge_model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        judge_model, dtype=torch.bfloat16 if dev.type != "cpu" else torch.float32
    ).to(dev).eval()
    gv2_sys = proto.QA_SYSTEM + " The answer field must contain exactly yes, no, unknown, or a single numeral from 0 to 9. Do not explain."

    @torch.no_grad()
    def _ask(pil_img, sys_p, user_p, max_tok):
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": sys_p}]},
            {"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": user_p}]},
        ]
        txt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[txt], images=[pil_img], return_tensors="pt").to(dev)
        out = model.generate(**inp, max_new_tokens=max_tok, do_sample=False)
        return proc.batch_decode(out[:, inp.input_ids.shape[1] :], skip_special_tokens=True)[0].strip()

    def _score_one(case, img_path: Path):
        with Image.open(img_path) as im:
            pil = im.convert("RGB")
        if case["benchmark"] == "geneval2":
            hits = []
            for q_text, exp_raw in case.get("vqa_list") or []:
                raw = _ask(pil, gv2_sys, q_text, 24)
                try:
                    ans = proto.interpret("geneval2", raw)
                except ValueError:
                    ans = None
                exp = proto.normalize("geneval2", str(exp_raw))
                hits.append(int(ans is not None and exp is not None and ans == exp))
            ok = int(all(h == 1 for h in hits)) if hits else 1
            return {"correct": ok, "qa_acc": round(sum(hits) / max(len(hits), 1), 4)}
        # geneval_pp
        user_text = (
            f"Instruction:\n{case['prompt']}\n\n"
            f"Expectation checklist:\n{pp_mod.metadata_to_checklist(case['benchmark_metadata'])}"
        )
        raw = _ask(pil, pp_mod.SYSTEM_PROMPTS["geneval_pp"], user_text, 192)
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I).strip()
        try:
            parsed = pp_mod.extract_result(cleaned)
            return {"correct": int(parsed["correct"])}
        except Exception:
            m = re.search(r'"correct"\s*:\s*(1|0|true|false|"1"|"0")', cleaned, flags=re.I)
            if m:
                return {"correct": int(m.group(1).strip('"').lower() in ("1", "true"))}
            return {"correct": None}

    return _score_one


def cmd_score(args):
    device = _setup_device(args.device)

    out_dir = Path(args.out).resolve()
    cases = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    case_by_id = {c["case_id"]: c for c in cases}
    arms_rows = {
        r["case_id"]: r
        for p in sorted((out_dir / "arms_shards").glob("arms_rank*.jsonl"))
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }

    sdir = out_dir / "score_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_path = sdir / f"score_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    _score_one = _load_judge(args.judge_model, device)
    ordered = [c["case_id"] for c in cases if c["case_id"] in arms_rows]
    mine = [cid for i, cid in enumerate(ordered) if i % int(args.world) == int(args.rank)]
    for pos, cid in enumerate(mine, 1):
        if cid in done:
            continue
        case = case_by_id[cid]
        ar = arms_rows[cid]
        cache = {}
        scores = {}
        for arm in ARMS:
            rel = ar["arms"][arm]["image_rel"]
            if rel not in cache:
                cache[rel] = _score_one(case, out_dir / rel)
            scores[arm] = cache[rel]
        rec = {"case_id": cid, "benchmark": case["benchmark"], "category": case["category"], "scores": scores}
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if pos == 1 or pos % 5 == 0 or pos == len(mine):
            print(
                f"[score r{args.rank:02d}] ({pos}/{len(mine)}) {cid} "
                f"A={scores['A_native']['correct']} B={scores['B_rewrite_resample']['correct']} "
                f"C={scores['C_reflective']['correct']} E={scores['E_shuffled_feedback']['correct']}",
                flush=True,
            )


def _paired_bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 20261123):
    if not deltas:
        return {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0}
    rng = random.Random(seed)
    n = len(deltas)
    mean = sum(deltas) / n
    boots = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": round(mean, 4), "ci95": [round(boots[int(0.025 * n_boot)], 4), round(boots[int(0.975 * n_boot)], 4)], "n": n}


def cmd_merge(args):
    out_dir = Path(args.out).resolve()
    payload = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))
    arms_rows = {
        r["case_id"]: r
        for p in sorted((out_dir / "arms_shards").glob("arms_rank*.jsonl"))
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }
    score_rows = {
        r["case_id"]: r
        for p in sorted((out_dir / "score_shards").glob("score_rank*.jsonl"))
        for l in p.read_text(encoding="utf-8").splitlines()
        if l.strip()
        for r in [json.loads(l)]
    }

    report = {"protocol": "reflective_t2i_v1_bench64", "expected": payload["total_cases"], "scored": len(score_rows), "benchmarks": {}}
    md_lines = [f"# `reflective_t2i_v1` 64-NPU Closed-Loop Causal Benchmark Summary (`{len(score_rows)}/{payload['total_cases']}`)", ""]

    for bench in ("geneval2", "geneval_pp"):
        ids = [
            c["case_id"]
            for c in payload["cases"]
            if c["benchmark"] == bench
            and c["case_id"] in score_rows
            and c["case_id"] in arms_rows
            and all(score_rows[c["case_id"]]["scores"][a]["correct"] is not None for a in ARMS)
        ]
        if not ids:
            continue
        trig_ids = [cid for cid in ids if arms_rows[cid]["outcome"] == "edit"]
        gated = Counter(arms_rows[cid]["gated_by"] for cid in ids if arms_rows[cid].get("gated_by"))
        arm_stats = {}
        md_lines.append(f"## {bench} (`n={len(ids)}`, triggered=`{len(trig_ids)}` [{len(trig_ids)/len(ids):.1%}], gated=`{dict(gated)}`)")
        md_lines.append("| Arm | Exact Acc ↑ | Atomic QA ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF | Mean Sec |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in ARMS:
            acc = sum(score_rows[cid]["scores"][arm]["correct"] for cid in ids) / len(ids)
            qa = (
                sum(score_rows[cid]["scores"][arm]["qa_acc"] for cid in ids) / len(ids)
                if bench == "geneval2"
                else None
            )
            tf = [cid for cid in trig_ids if score_rows[cid]["scores"]["A_native"]["correct"] == 0]
            tp = [cid for cid in trig_ids if score_rows[cid]["scores"]["A_native"]["correct"] == 1]
            fix = sum(score_rows[cid]["scores"][arm]["correct"] == 1 for cid in tf)
            dmg = sum(score_rows[cid]["scores"][arm]["correct"] == 0 for cid in tp)
            hf = sum(arms_rows[cid]["arms"][arm]["hf_grad"] for cid in ids) / len(ids)
            sec = sum(arms_rows[cid]["arms"][arm]["seconds"] for cid in ids) / len(ids)
            arm_stats[arm] = {
                "acc": round(acc, 4),
                "qa_acc": round(qa, 4) if qa is not None else None,
                "fixed": fix,
                "trig_fail": len(tf),
                "fix_rate": round(fix / max(len(tf), 1), 4),
                "damaged": dmg,
                "trig_pass": len(tp),
                "damage_rate": round(dmg / max(len(tp), 1), 4),
                "net_fixed": fix - dmg,
                "mean_hf": round(hf, 5),
                "mean_sec": round(sec, 2),
            }
            qa_str = f"{qa:.4f}" if qa is not None else "—"
            md_lines.append(
                f"| `{arm}` | **{acc:.4f}** | {qa_str} | {fix}/{len(tf)} ({fix/max(len(tf),1):.1%}) | "
                f"{dmg}/{len(tp)} ({dmg/max(len(tp),1):.1%}) | {fix-dmg:+d} | {hf:.5f} | {sec:.2f}s |"
            )

        # Paired causal bootstrap comparisons
        c_vs_a = _paired_bootstrap_ci([score_rows[cid]["scores"]["C_reflective"]["correct"] - score_rows[cid]["scores"]["A_native"]["correct"] for cid in ids])
        c_vs_e = _paired_bootstrap_ci([score_rows[cid]["scores"]["C_reflective"]["correct"] - score_rows[cid]["scores"]["E_shuffled_feedback"]["correct"] for cid in ids])
        c_vs_b = _paired_bootstrap_ci([score_rows[cid]["scores"]["C_reflective"]["correct"] - score_rows[cid]["scores"]["B_rewrite_resample"]["correct"] for cid in ids])
        md_lines.append(
            f"- Paired Δ Exact Acc (95% CI): `C - A = {c_vs_a['mean']:+.4f} {c_vs_a['ci95']}` | "
            f"`C - B = {c_vs_b['mean']:+.4f} {c_vs_b['ci95']}` | `C - E (causal) = {c_vs_e['mean']:+.4f} {c_vs_e['ci95']}`"
        )

        # Action breakdown on triggered cases
        act_breakdown = {}
        md_lines.append("")
        md_lines.append("| Action (triggered) | n | A_native Acc | B_rewrite Acc | C_reflective Acc | E_shuffled Acc | C Fix / TF | C Dmg / TP | C Net |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for act in ("add", "remove", "attr", "move"):
            aids = [cid for cid in trig_ids if arms_rows[cid]["action"] == act]
            if not aids:
                continue
            tf = [cid for cid in aids if score_rows[cid]["scores"]["A_native"]["correct"] == 0]
            tp = [cid for cid in aids if score_rows[cid]["scores"]["A_native"]["correct"] == 1]
            cfix = sum(score_rows[cid]["scores"]["C_reflective"]["correct"] == 1 for cid in tf)
            cdmg = sum(score_rows[cid]["scores"]["C_reflective"]["correct"] == 0 for cid in tp)
            accs = {a: round(sum(score_rows[cid]["scores"][a]["correct"] for cid in aids) / len(aids), 4) for a in ARMS}
            act_breakdown[act] = {"n": len(aids), **accs, "c_fix": cfix, "tf": len(tf), "c_dmg": cdmg, "tp": len(tp), "c_net": cfix - cdmg}
            md_lines.append(
                f"| `{act}` | {len(aids)} | {accs['A_native']:.4f} | {accs['B_rewrite_resample']:.4f} | "
                f"**{accs['C_reflective']:.4f}** | {accs['E_shuffled_feedback']:.4f} | "
                f"{cfix}/{len(tf)} | {cdmg}/{len(tp)} | {cfix-cdmg:+d} |"
            )
        md_lines.append("")
        report["benchmarks"][bench] = {
            "n": len(ids),
            "triggered": len(trig_ids),
            "gated": dict(gated),
            "arms": arm_stats,
            "paired_delta": {"C_minus_A": c_vs_a, "C_minus_B": c_vs_b, "C_minus_E": c_vs_e},
            "by_action": act_breakdown,
        }

    md = "\n".join(md_lines) + "\n"
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(md, flush=True)


def _run_phase_64(phase: str, worker_py: str, extra_args: list[str], out_dir: Path):
    hosts = [h.strip() for h in os.environ.get("VC_WORKER_HOSTS", "").split(",") if h.strip()]
    if len(hosts) != 4:
        raise ValueError(f"Expected 4 hosts in VC_WORKER_HOSTS, got {len(hosts)}")
    world = 4 * PER_NODE
    log_dir = out_dir / f"{phase}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    helper = ROOT / "scripts/start_train_node.py"

    procs = []
    for node_idx, host in enumerate(hosts):
        lines = [
            "set -euo pipefail",
            f"cd {shlex.quote(str(ROOT))}",
            f"export PYTHONPATH={shlex.quote(f'{ROOT}:{DEFAULT_ZIMAGE_CORE}:{DEFAULT_SD35_DEPS}')}",
            "export PYTHONUNBUFFERED=1",
            "export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "export OMP_NUM_THREADS=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
            "PIDS=()",
        ]
        for lr in range(PER_NODE):
            gr = node_idx * PER_NODE + lr
            cmd = [
                worker_py, str(ROOT / "scripts/bench_reflective_t2i64.py"), phase,
                "--out", str(out_dir), "--rank", str(gr), "--world", str(world), "--device", f"npu:{lr}",
                *extra_args,
            ]
            lines.append(f"{shlex.join(cmd)} > {shlex.quote(str(log_dir / f'rank_{gr:02d}.log'))} 2>&1 &")
            lines.append("PIDS+=($!)")
            if lr + 1 < PER_NODE:
                lines.append("sleep 0.25")
        lines.append('for pid in "${PIDS[@]}"; do wait "$pid"; done')
        lines.append(f"echo NODE_{node_idx}_{phase}_DONE")
        script = "\n".join(lines)
        if node_idx == 0:
            full_cmd = ["bash", "-lc", script]
        else:
            full_cmd = [
                "ssh", "-n", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                "root@" + host, "bash", "-lc", shlex.quote(script),
            ]
        procs.append((node_idx, subprocess.Popen(full_cmd)))
    for node_idx, proc in procs:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Phase {phase} failed on node {node_idx} with exit code {rc}")
    print(f"=== PHASE {phase} COMPLETED ON 64 NPUs ===", flush=True)


def cmd_run_all64(args):
    out_dir = Path(args.out).resolve()
    cmd_prepare(args)
    _run_phase_64("gen-prefix", args.u15_py, ["--config", str(Path(args.config).resolve())], out_dir)
    _run_phase_64("reflect", args.score_py, ["--config", str(Path(args.config).resolve()), "--observer-model", args.observer_model], out_dir)
    _run_phase_64("gen-arms", args.u15_py, ["--config", str(Path(args.config).resolve()), "--checkpoint", str(Path(args.checkpoint).resolve())], out_dir)
    _run_phase_64("score", args.score_py, ["--judge-model", args.judge_model], out_dir)
    cmd_merge(args)


def _load_jsonl_dir(directory: Path, pattern: str) -> dict[str, dict]:
    rows = {}
    for p in sorted(directory.glob(pattern)):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["case_id"]] = r
    return rows


def cmd_gen_r1_imp(args):
    """Generate Round-1 concise-imperative reflective outputs (C_r1_imperative) on triggered cases."""
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = _setup_device(args.device)
    adapter = _build_generator(cfg, device, attach_corrector=True)
    sd = torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True)
    adapter.corrector.load_state_dict_trainable(sd, strict=True)

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out).resolve()
    cases = json.loads((base_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    prefix_rows = _load_jsonl_dir(base_dir / "prefix_shards", "prefix_rank*.jsonl")
    ref_rows = _load_jsonl_dir(base_dir / "reflect_shards", "reflect_rank*.jsonl")

    gcfg, rcfg, ccfg = cfg["generator"], cfg["reflection"], cfg["corrector"]
    p0 = float(rcfg["read_progress"])
    p_stop = float(rcfg["stop_progress"])
    gscale = float(gcfg.get("guidance_scale", 4.0))
    cfg_int = tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80)))
    cond_mode = ccfg.get("condition_mode", "feedback_only")
    transition = IsochronicTransitionV2(
        TransitionConfig(
            read_progress=p0,
            stop_progress=p_stop,
            anchor_mode=ccfg.get("anchor_mode", "conditional_native"),
            edit_cfg_scale=float(ccfg.get("edit_cfg_scale", 1.4)),
            trust_region_rho=float(ccfg.get("trust_region_rho", 0.65)),
            noise_refresh=float(ccfg.get("noise_refresh", 0.0)),
        )
    )
    null_cond = adapter.encode_prompt(None)

    sdir = out_dir / "r1_imp_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_path = sdir / f"r1_imp_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    triggered_cases = [c for c in cases if ref_rows[c["case_id"]]["outcome"] == "edit" and ref_rows[c["case_id"]]["instruction"]]
    mine = [c for i, c in enumerate(triggered_cases) if i % int(args.world) == int(args.rank)]
    with torch.no_grad():
        for pos, case in enumerate(mine, 1):
            cid = case["case_id"]
            if cid in done:
                continue
            pr = prefix_rows[cid]
            rr = ref_rows[cid]
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", cid)
            cdir = out_dir / "images" / case["benchmark"] / safe
            cdir.mkdir(parents=True, exist_ok=True)

            seed = int(case["seed"])
            prompt = case["prompt"]
            orig_cond = adapter.encode_prompt(prompt)
            z0 = adapter.sample_initial_noise(seed=seed)
            state_p0, cached_clean = adapter.sample_until(
                z0, p0, orig_cond, null_cond, gscale, cfg_interval=cfg_int, return_clean_at_end=True
            )
            _sync(device)
            t0 = time.perf_counter()
            imp = compile_concise_imperative(
                rr["instruction"],
                action=rr["action"],
                coarse_region_hint=rr.get("coarse_region_hint") or "",
            )
            edit_cond = adapter.encode_edit_condition(
                feedback=imp,
                original_prompt=prompt,
                target_rewrite=rr.get("target_rewrite") or prompt,
                condition_mode=cond_mode,
            )
            tr = transition.apply(
                adapter,
                state_p0,
                edit_cond,
                orig_cond,
                null_cond,
                cached_native_clean=cached_clean,
                generator=torch.Generator(device=device).manual_seed(seed * 97),
            )
            lat = adapter.continue_native(tr.state, orig_cond, null_cond, gscale, cfg_interval=cfg_int)
            img = adapter.decode_preview(lat)
            _sync(device)
            sec = time.perf_counter() - t0
            img_path = cdir / "C_r1_imperative.png"
            img.save(img_path)
            rec = {
                "case_id": cid,
                "benchmark": case["benchmark"],
                "action": rr["action"],
                "compiled_instruction": imp,
                "image_rel": str(img_path.relative_to(out_dir)),
                "hf_grad": _pil_hf_grad(img),
                "seconds": round(pr["sec_prefix"] + pr["sec_preview"] + rr["sec_reflect"] + sec, 4),
            }
            with shard_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pos == 1 or pos % 5 == 0 or pos == len(mine):
                print(f"[r1_imp r{args.rank:02d}] ({pos}/{len(mine)}) {cid} imp={imp[:60]!r}", flush=True)


def cmd_reflect_r2(args):
    """Run Round-2 external VLM Observer re-reflection on C_r1_imperative outputs."""
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rcfg = cfg["reflection"]
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out).resolve()
    cases = json.loads((base_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    r1_rows = _load_jsonl_dir(out_dir / "r1_imp_shards", "r1_imp_rank*.jsonl")

    targets = [c for c in cases if c["case_id"] in r1_rows]
    mine = [c for i, c in enumerate(targets) if i % int(args.world) == int(args.rank)]
    obs = LocalHFObserver(args.observer_model, device=args.device)
    policy = ReflectionPolicy(
        ReflectionPolicyConfig(
            focused_recheck_actions=tuple(rcfg.get("focused_recheck_actions", ("add", "remove"))),
            confidence_threshold=float(rcfg.get("confidence_threshold", 0.75)),
            contradiction_gate=bool(rcfg.get("contradiction_gate", True)),
        )
    )

    sdir = out_dir / "reflect_r2_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_path = sdir / f"reflect_r2_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    for pos, case in enumerate(mine, 1):
        cid = case["case_id"]
        if cid in done:
            continue
        t0 = time.perf_counter()
        with Image.open(out_dir / r1_rows[cid]["image_rel"]) as im:
            dec = policy.reflect(obs, im.convert("RGB"), case["prompt"])
        sec = round(time.perf_counter() - t0, 4)
        rec = {
            "case_id": cid,
            "benchmark": case["benchmark"],
            "outcome": dec.outcome,
            "action": dec.action,
            "instruction": dec.instruction,
            "target_rewrite": dec.target_rewrite,
            "coarse_region_hint": dec.coarse_region_hint,
            "confidence": dec.confidence,
            "gated_by": dec.gated_by,
            "sec_reflect_r2": sec,
        }
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if pos == 1 or pos % 5 == 0 or pos == len(mine):
            print(f"[reflect_r2 r{args.rank:02d}] ({pos}/{len(mine)}) {cid} -> {dec.outcome} ({dec.action}) {sec:.1f}s", flush=True)


def cmd_gen_r2(args):
    """Apply Round-2 atomic isochronic correction on cases where Round-2 Observer still flags an edit."""
    import numpy as np

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = _setup_device(args.device)
    adapter = _build_generator(cfg, device, attach_corrector=True)
    sd = torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True)
    adapter.corrector.load_state_dict_trainable(sd, strict=True)

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out).resolve()
    cases = json.loads((base_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    r1_rows = _load_jsonl_dir(out_dir / "r1_imp_shards", "r1_imp_rank*.jsonl")
    r2_ref = _load_jsonl_dir(out_dir / "reflect_r2_shards", "reflect_r2_rank*.jsonl")

    gcfg, ccfg = cfg["generator"], cfg["corrector"]
    p0_r2 = float(args.r2_read_progress)
    p_stop_r2 = float(args.r2_stop_progress)
    gscale = float(gcfg.get("guidance_scale", 4.0))
    cfg_int = tuple(float(v) for v in gcfg.get("cfg_interval", (0.0, 0.80)))
    cond_mode = ccfg.get("condition_mode", "feedback_only")
    transition = IsochronicTransitionV2(
        TransitionConfig(
            read_progress=p0_r2,
            stop_progress=p_stop_r2,
            anchor_mode=ccfg.get("anchor_mode", "conditional_native"),
            edit_cfg_scale=float(ccfg.get("edit_cfg_scale", 1.4)),
            trust_region_rho=float(ccfg.get("trust_region_rho", 0.65)),
            noise_refresh=float(ccfg.get("noise_refresh", 0.0)),
        )
    )
    null_cond = adapter.encode_prompt(None)

    sdir = out_dir / "r2_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_path = sdir / f"r2_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    targets = [c for c in cases if c["case_id"] in r1_rows]
    mine = [c for i, c in enumerate(targets) if i % int(args.world) == int(args.rank)]
    with torch.no_grad():
        for pos, case in enumerate(mine, 1):
            cid = case["case_id"]
            if cid in done:
                continue
            r1 = r1_rows[cid]
            rr2 = r2_ref[cid]
            r2_trig = rr2["outcome"] == "edit" and bool(rr2["instruction"])
            if not r2_trig:
                rec = {
                    "case_id": cid,
                    "benchmark": case["benchmark"],
                    "r2_triggered": False,
                    "r2_action": "none",
                    "r2_instruction": "",
                    "image_rel": r1["image_rel"],
                    "hf_grad": r1["hf_grad"],
                    "seconds": round(r1["seconds"] + rr2["sec_reflect_r2"], 4),
                }
            else:
                _sync(device)
                t0 = time.perf_counter()
                seed = int(case["seed"])
                prompt = case["prompt"]
                orig_cond = adapter.encode_prompt(prompt)
                with Image.open(out_dir / r1["image_rel"]) as im:
                    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
                rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
                x_r1 = adapter.encode_clean(rgb)
                eps = adapter.sample_initial_noise(seed=seed * 131 + 7)
                z_r2 = adapter.geometry.interpolate(x_r1, eps, p0_r2)
                state_r2 = LatentState(tensor=z_r2, progress=p0_r2)

                imp2 = compile_concise_imperative(
                    rr2["instruction"],
                    action=rr2["action"],
                    coarse_region_hint=rr2.get("coarse_region_hint") or "",
                )
                edit_cond2 = adapter.encode_edit_condition(
                    feedback=imp2,
                    original_prompt=prompt,
                    target_rewrite=rr2.get("target_rewrite") or prompt,
                    condition_mode=cond_mode,
                )
                tr2 = transition.apply(
                    adapter,
                    state_r2,
                    edit_cond2,
                    orig_cond,
                    null_cond,
                    cached_native_clean=None,
                    generator=torch.Generator(device=device).manual_seed(seed * 193),
                )
                lat2 = adapter.continue_native(tr2.state, orig_cond, null_cond, gscale, cfg_interval=cfg_int)
                img2 = adapter.decode_preview(lat2)
                _sync(device)
                sec2 = time.perf_counter() - t0
                img_path2 = (out_dir / r1["image_rel"]).parent / "C_r2_atomic.png"
                img2.save(img_path2)
                rec = {
                    "case_id": cid,
                    "benchmark": case["benchmark"],
                    "r2_triggered": True,
                    "r2_action": rr2["action"],
                    "r2_instruction": imp2,
                    "image_rel": str(img_path2.relative_to(out_dir)),
                    "hf_grad": _pil_hf_grad(img2),
                    "seconds": round(r1["seconds"] + rr2["sec_reflect_r2"] + sec2, 4),
                }
            with shard_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if pos == 1 or pos % 5 == 0 or pos == len(mine):
                print(f"[r2 r{args.rank:02d}] ({pos}/{len(mine)}) {cid} r2_trig={r2_trig} ({rec['r2_action']})", flush=True)


def cmd_score_iter(args):
    """Score C_r1_imperative and C_r2_atomic on triggered cases with independent Qwen3-VL-8B-Instruct."""
    device = _setup_device(args.device)

    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out).resolve()
    cases = json.loads((base_dir / "cases.json").read_text(encoding="utf-8"))["cases"]
    case_by_id = {c["case_id"]: c for c in cases}
    r1_rows = _load_jsonl_dir(out_dir / "r1_imp_shards", "r1_imp_rank*.jsonl")
    r2_rows = _load_jsonl_dir(out_dir / "r2_shards", "r2_rank*.jsonl")

    sdir = out_dir / "score_iter_shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_path = sdir / f"score_iter_rank{int(args.rank):02d}.jsonl"
    done = set()
    if shard_path.exists():
        for l in shard_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["case_id"])

    _score_one = _load_judge(args.judge_model, device)
    ordered = [c["case_id"] for c in cases if c["case_id"] in r1_rows and c["case_id"] in r2_rows]
    mine = [cid for i, cid in enumerate(ordered) if i % int(args.world) == int(args.rank)]
    for pos, cid in enumerate(mine, 1):
        if cid in done:
            continue
        case = case_by_id[cid]
        rel1 = r1_rows[cid]["image_rel"]
        rel2 = r2_rows[cid]["image_rel"]
        sc1 = _score_one(case, out_dir / rel1)
        sc2 = sc1 if rel2 == rel1 else _score_one(case, out_dir / rel2)
        rec = {"case_id": cid, "benchmark": case["benchmark"], "scores": {"C_r1_imperative": sc1, "C_r2_atomic": sc2}}
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if pos == 1 or pos % 5 == 0 or pos == len(mine):
            print(f"[score_iter r{args.rank:02d}] ({pos}/{len(mine)}) {cid} R1={sc1['correct']} R2={sc2['correct']}", flush=True)


def cmd_merge_iter(args):
    """Merge 6-arm apples-to-apples comparison: A, E, C_r1_verbose, C_r1_imperative, C_r2_atomic, B."""
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out).resolve()
    payload = json.loads((base_dir / "cases.json").read_text(encoding="utf-8"))
    base_arms = _load_jsonl_dir(base_dir / "arms_shards", "arms_rank*.jsonl")
    base_scores = _load_jsonl_dir(base_dir / "score_shards", "score_rank*.jsonl")
    r1_rows = _load_jsonl_dir(out_dir / "r1_imp_shards", "r1_imp_rank*.jsonl")
    r2_rows = _load_jsonl_dir(out_dir / "r2_shards", "r2_rank*.jsonl")
    iter_scores = _load_jsonl_dir(out_dir / "score_iter_shards", "score_iter_rank*.jsonl")

    all_arms = ("A_native", "E_shuffled_feedback", "C_reflective", "C_r1_imperative", "C_r2_atomic", "B_rewrite_resample")
    report = {"protocol": "reflective_t2i_v1_iter64", "benchmarks": {}}
    md_lines = ["# `reflective_t2i_v1` 64-NPU Imperative + 2-Round Iterative Self-Reflection Benchmark (`1080/1080`)", ""]

    for bench in ("geneval2", "geneval_pp"):
        ids = []
        merged_scores = {}
        for c in payload["cases"]:
            cid = c["case_id"]
            if c["benchmark"] != bench or cid not in base_scores or cid not in base_arms:
                continue
            sc = dict(base_scores[cid]["scores"])
            if cid in iter_scores:
                sc["C_r1_imperative"] = iter_scores[cid]["scores"]["C_r1_imperative"]
                sc["C_r2_atomic"] = iter_scores[cid]["scores"]["C_r2_atomic"]
            else:
                sc["C_r1_imperative"] = sc["A_native"]
                sc["C_r2_atomic"] = sc["A_native"]
            if all(sc[a]["correct"] is not None for a in all_arms):
                ids.append(cid)
                merged_scores[cid] = sc
        if not ids:
            continue

        trig_ids = [cid for cid in ids if base_arms[cid]["outcome"] == "edit"]
        r2_trig_ids = [cid for cid in trig_ids if cid in r2_rows and r2_rows[cid]["r2_triggered"]]
        md_lines.append(f"## {bench} (`n={len(ids)}`, R1 triggered=`{len(trig_ids)}`, R2 triggered=`{len(r2_trig_ids)}`)")
        md_lines.append("| Arm | Exact Acc ↑ | Atomic QA ↑ | Fix / TrigFail ↑ | Damage / TrigPass ↓ | Net Fixed | Mean HF |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        arm_stats = {}
        for arm in all_arms:
            acc = sum(merged_scores[cid][arm]["correct"] for cid in ids) / len(ids)
            qa = sum(merged_scores[cid][arm]["qa_acc"] for cid in ids) / len(ids) if bench == "geneval2" else None
            tf = [cid for cid in trig_ids if merged_scores[cid]["A_native"]["correct"] == 0]
            tp = [cid for cid in trig_ids if merged_scores[cid]["A_native"]["correct"] == 1]
            fix = sum(merged_scores[cid][arm]["correct"] == 1 for cid in tf)
            dmg = sum(merged_scores[cid][arm]["correct"] == 0 for cid in tp)
            hfs = []
            for cid in ids:
                if arm == "C_r1_imperative" and cid in r1_rows:
                    hfs.append(r1_rows[cid]["hf_grad"])
                elif arm == "C_r2_atomic" and cid in r2_rows:
                    hfs.append(r2_rows[cid]["hf_grad"])
                elif arm in base_arms[cid]["arms"]:
                    hfs.append(base_arms[cid]["arms"][arm]["hf_grad"])
                else:
                    hfs.append(base_arms[cid]["arms"]["A_native"]["hf_grad"])
            hf = sum(hfs) / len(hfs)
            arm_stats[arm] = {
                "acc": round(acc, 4),
                "qa_acc": round(qa, 4) if qa is not None else None,
                "fixed": fix, "trig_fail": len(tf), "fix_rate": round(fix / max(len(tf), 1), 4),
                "damaged": dmg, "trig_pass": len(tp), "damage_rate": round(dmg / max(len(tp), 1), 4),
                "net_fixed": fix - dmg, "mean_hf": round(hf, 5),
            }
            qa_str = f"{qa:.4f}" if qa is not None else "—"
            md_lines.append(
                f"| `{arm}` | **{acc:.4f}** | {qa_str} | {fix}/{len(tf)} ({fix/max(len(tf),1):.1%}) | "
                f"{dmg}/{len(tp)} ({dmg/max(len(tp),1):.1%}) | {fix-dmg:+d} | {hf:.5f} |"
            )

        r1_vs_a = _paired_bootstrap_ci([merged_scores[cid]["C_r1_imperative"]["correct"] - merged_scores[cid]["A_native"]["correct"] for cid in ids])
        r2_vs_a = _paired_bootstrap_ci([merged_scores[cid]["C_r2_atomic"]["correct"] - merged_scores[cid]["A_native"]["correct"] for cid in ids])
        r2_vs_e = _paired_bootstrap_ci([merged_scores[cid]["C_r2_atomic"]["correct"] - merged_scores[cid]["E_shuffled_feedback"]["correct"] for cid in ids])
        r2_vs_b = _paired_bootstrap_ci([merged_scores[cid]["C_r2_atomic"]["correct"] - merged_scores[cid]["B_rewrite_resample"]["correct"] for cid in ids])
        r2_vs_r1 = _paired_bootstrap_ci([merged_scores[cid]["C_r2_atomic"]["correct"] - merged_scores[cid]["C_r1_imperative"]["correct"] for cid in ids])
        md_lines.append(
            f"- Paired Δ (95% CI): `R1_imp - A = {r1_vs_a['mean']:+.4f} {r1_vs_a['ci95']}` | "
            f"`R2 - A = {r2_vs_a['mean']:+.4f} {r2_vs_a['ci95']}` | `R2 - E (causal) = {r2_vs_e['mean']:+.4f} {r2_vs_e['ci95']}` | "
            f"`R2 - B = {r2_vs_b['mean']:+.4f} {r2_vs_b['ci95']}` | `R2 - R1_imp = {r2_vs_r1['mean']:+.4f} {r2_vs_r1['ci95']}`"
        )
        md_lines.append("")
        md_lines.append("| Action (R1 triggered) | n | A_native | E_shuf | C_verbose | C_r1_imp | C_r2_atomic | B_rewrite |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        by_act = {}
        for act in ("add", "remove", "attr", "move"):
            aids = [cid for cid in trig_ids if base_arms[cid]["action"] == act]
            if not aids:
                continue
            accs = {a: round(sum(merged_scores[cid][a]["correct"] for cid in aids) / len(aids), 4) for a in all_arms}
            by_act[act] = {"n": len(aids), **accs}
            md_lines.append(
                f"| `{act}` | {len(aids)} | {accs['A_native']:.4f} | {accs['E_shuffled_feedback']:.4f} | "
                f"{accs['C_reflective']:.4f} | {accs['C_r1_imperative']:.4f} | **{accs['C_r2_atomic']:.4f}** | {accs['B_rewrite_resample']:.4f} |"
            )
        md_lines.append("")
        report["benchmarks"][bench] = {
            "n": len(ids), "r1_triggered": len(trig_ids), "r2_triggered": len(r2_trig_ids),
            "arms": arm_stats,
            "paired_delta": {"R1_minus_A": r1_vs_a, "R2_minus_A": r2_vs_a, "R2_minus_E": r2_vs_e, "R2_minus_B": r2_vs_b, "R2_minus_R1": r2_vs_r1},
            "by_action": by_act,
        }

    md = "\n".join(md_lines) + "\n"
    (out_dir / "summary_iter.md").write_text(md, encoding="utf-8")
    (out_dir / "summary_iter.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(md, flush=True)


def cmd_run_iter64(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = str(Path(args.base_dir).resolve())
    cfg = str(Path(args.config).resolve())
    ckpt = str(Path(args.checkpoint).resolve())
    _run_phase_64("gen-r1-imp", args.u15_py, ["--config", cfg, "--checkpoint", ckpt, "--base-dir", base], out_dir)
    _run_phase_64("reflect-r2", args.score_py, ["--config", cfg, "--base-dir", base, "--observer-model", args.observer_model], out_dir)
    _run_phase_64("gen-r2", args.u15_py, ["--config", cfg, "--checkpoint", ckpt, "--base-dir", base, "--r2-read-progress", str(args.r2_read_progress), "--r2-stop-progress", str(args.r2_stop_progress)], out_dir)
    _run_phase_64("score-iter", args.score_py, ["--base-dir", base, "--judge-model", args.judge_model], out_dir)
    cmd_merge_iter(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--out", required=True)
    sp.add_argument("--geneval2", default=DEFAULT_GENEVAL2)
    sp.add_argument("--geneval-pp", default=DEFAULT_GENEVAL_PP)
    sp.add_argument("--skip-geneval2", action="store_true")
    sp.add_argument("--skip-geneval-pp", action="store_true")
    sp.add_argument("--limit-per-bench", type=int, default=0)
    sp.set_defaults(func=cmd_prepare)

    sgp = sub.add_parser("gen-prefix")
    sgp.add_argument("--config", required=True)
    sgp.add_argument("--out", required=True)
    sgp.add_argument("--rank", type=int, required=True)
    sgp.add_argument("--world", type=int, required=True)
    sgp.add_argument("--device", default="npu:0")
    sgp.set_defaults(func=cmd_gen_prefix)

    sr = sub.add_parser("reflect")
    sr.add_argument("--config", required=True)
    sr.add_argument("--out", required=True)
    sr.add_argument("--observer-model", default=DEFAULT_VLM_MODEL)
    sr.add_argument("--rank", type=int, required=True)
    sr.add_argument("--world", type=int, required=True)
    sr.add_argument("--device", default="npu:0")
    sr.set_defaults(func=cmd_reflect)

    sga = sub.add_parser("gen-arms")
    sga.add_argument("--config", required=True)
    sga.add_argument("--checkpoint", required=True)
    sga.add_argument("--out", required=True)
    sga.add_argument("--rank", type=int, required=True)
    sga.add_argument("--world", type=int, required=True)
    sga.add_argument("--device", default="npu:0")
    sga.set_defaults(func=cmd_gen_arms)

    ss = sub.add_parser("score")
    ss.add_argument("--out", required=True)
    ss.add_argument("--judge-model", default=DEFAULT_VLM_MODEL)
    ss.add_argument("--rank", type=int, required=True)
    ss.add_argument("--world", type=int, required=True)
    ss.add_argument("--device", default="npu:0")
    ss.set_defaults(func=cmd_score)

    sm = sub.add_parser("merge")
    sm.add_argument("--out", required=True)
    sm.set_defaults(func=cmd_merge)

    sall = sub.add_parser("run-all64")
    sall.add_argument("--config", required=True)
    sall.add_argument("--checkpoint", required=True)
    sall.add_argument("--out", required=True)
    sall.add_argument("--geneval2", default=DEFAULT_GENEVAL2)
    sall.add_argument("--geneval-pp", default=DEFAULT_GENEVAL_PP)
    sall.add_argument("--skip-geneval2", action="store_true")
    sall.add_argument("--skip-geneval-pp", action="store_true")
    sall.add_argument("--observer-model", default=DEFAULT_VLM_MODEL)
    sall.add_argument("--judge-model", default=DEFAULT_VLM_MODEL)
    sall.add_argument("--u15-py", default=DEFAULT_U15_PY)
    sall.add_argument("--score-py", default=DEFAULT_SCORE_PY)
    sall.set_defaults(func=cmd_run_all64)

    sr1 = sub.add_parser("gen-r1-imp")
    sr1.add_argument("--config", required=True)
    sr1.add_argument("--checkpoint", required=True)
    sr1.add_argument("--base-dir", required=True)
    sr1.add_argument("--out", required=True)
    sr1.add_argument("--rank", type=int, required=True)
    sr1.add_argument("--world", type=int, required=True)
    sr1.add_argument("--device", default="npu:0")
    sr1.set_defaults(func=cmd_gen_r1_imp)

    srf2 = sub.add_parser("reflect-r2")
    srf2.add_argument("--config", required=True)
    srf2.add_argument("--base-dir", required=True)
    srf2.add_argument("--out", required=True)
    srf2.add_argument("--observer-model", default=DEFAULT_VLM_MODEL)
    srf2.add_argument("--rank", type=int, required=True)
    srf2.add_argument("--world", type=int, required=True)
    srf2.add_argument("--device", default="npu:0")
    srf2.set_defaults(func=cmd_reflect_r2)

    sr2 = sub.add_parser("gen-r2")
    sr2.add_argument("--config", required=True)
    sr2.add_argument("--checkpoint", required=True)
    sr2.add_argument("--base-dir", required=True)
    sr2.add_argument("--out", required=True)
    sr2.add_argument("--r2-read-progress", type=float, default=0.34)
    sr2.add_argument("--r2-stop-progress", type=float, default=0.58)
    sr2.add_argument("--rank", type=int, required=True)
    sr2.add_argument("--world", type=int, required=True)
    sr2.add_argument("--device", default="npu:0")
    sr2.set_defaults(func=cmd_gen_r2)

    ssi = sub.add_parser("score-iter")
    ssi.add_argument("--base-dir", required=True)
    ssi.add_argument("--out", required=True)
    ssi.add_argument("--judge-model", default=DEFAULT_VLM_MODEL)
    ssi.add_argument("--rank", type=int, required=True)
    ssi.add_argument("--world", type=int, required=True)
    ssi.add_argument("--device", default="npu:0")
    ssi.set_defaults(func=cmd_score_iter)

    smi = sub.add_parser("merge-iter")
    smi.add_argument("--base-dir", required=True)
    smi.add_argument("--out", required=True)
    smi.set_defaults(func=cmd_merge_iter)

    siter = sub.add_parser("run-iter64")
    siter.add_argument("--config", required=True)
    siter.add_argument("--checkpoint", required=True)
    siter.add_argument("--base-dir", required=True)
    siter.add_argument("--out", required=True)
    siter.add_argument("--r2-read-progress", type=float, default=0.34)
    siter.add_argument("--r2-stop-progress", type=float, default=0.58)
    siter.add_argument("--observer-model", default=DEFAULT_VLM_MODEL)
    siter.add_argument("--judge-model", default=DEFAULT_VLM_MODEL)
    siter.add_argument("--u15-py", default=DEFAULT_U15_PY)
    siter.add_argument("--score-py", default=DEFAULT_SCORE_PY)
    siter.set_defaults(func=cmd_run_iter64)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
