#!/usr/bin/env python3
"""Run GLM-5.3-Flash reflection (DESCRIBE -> JUDGE + Breakthrough-A gates) on cached 4-step previews."""

import argparse
import base64
import glob
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_latent_cot.count_spatial.glm import GlmClient, resolve_glm_endpoint, resolve_glm_key
from qwen_latent_cot.u15_target_correction.diagnosis import (
    DESCRIBE_SYSTEM,
    DESCRIBE_TEMPLATE,
    FOCUSED_SCAN_TEMPLATE,
    JUDGE_SYSTEM,
    JUDGE_TEMPLATE,
    check_instruction_consistency,
    extract_add_subject,
    parse_reflection,
)


def classify_diagnosed_action(instruction):
    low = " ".join(str(instruction or "").lower().split())
    if low.startswith("add "):
        return "add"
    if low.startswith("remove "):
        return "remove"
    if low.startswith(("move ", "relocate ", "shift ", "place ")):
        return "move"
    if low.startswith(("change ", "replace ", "recolor ", "make ", "turn ")):
        return "attr"
    return "other"


def _encode_png(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _vision_msg(sys_msg, prompt_text, b64_img):
    return [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ],
        },
    ]


def _text_msg(sys_msg, prompt_text):
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt_text},
    ]


def reflect_one(client, rec, prev_bench_dir):
    cid = rec["case_id"]
    prompt = rec["prompt"]
    base_rel = rec["arms"]["baseline"]["image_rel"]
    prev_path = Path(prev_bench_dir) / Path(base_rel).parent / "preview.png"
    b64 = _encode_png(prev_path)

    desc = client._http_transport(
        _vision_msg(DESCRIBE_SYSTEM, DESCRIBE_TEMPLATE, b64),
        max_tokens=512,
    ).strip()

    judg = client._http_transport(
        _text_msg(JUDGE_SYSTEM, JUDGE_TEMPLATE.format(prompt=prompt, description=desc)),
        max_tokens=384,
    ).strip()
    parsed = parse_reflection(judg, desc)
    initial_instr = parsed["instruction"]

    if parsed["outcome"] == "edit":
        subj = extract_add_subject(parsed["instruction"])
        if subj:
            fscan = client._http_transport(
                _vision_msg(DESCRIBE_SYSTEM, FOCUSED_SCAN_TEMPLATE.format(subject=subj), b64),
                max_tokens=256,
            ).strip()
            comb = desc + f"\n\nFocused re-scan for '{subj}':\n" + fscan
            judg2 = client._http_transport(
                _text_msg(JUDGE_SYSTEM, JUDGE_TEMPLATE.format(prompt=prompt, description=comb)),
                max_tokens=384,
            ).strip()
            reparsed = parse_reflection(judg2, comb)
            if reparsed["outcome"] not in ("failed_parse", "failed_unconverted"):
                if reparsed["outcome"] == "satisfied":
                    reparsed["gated_by"] = "focused_add_recheck"
                    reparsed["gated_instruction"] = initial_instr
                parsed = reparsed
                desc = comb
                judg = judg2
            parsed["rechecked_add_subject"] = subj

    if parsed["outcome"] == "edit":
        ok, reason = check_instruction_consistency(parsed["instruction"], prompt)
        if not ok:
            parsed["gated_instruction"] = parsed["instruction"]
            parsed["instruction"] = ""
            parsed["match"] = True
            parsed["outcome"] = "satisfied"
            parsed["gated_by"] = reason

    outcome = str(parsed["outcome"])
    instr = " ".join(str(parsed.get("instruction") or "").split())
    act = classify_diagnosed_action(instr) if outcome == "edit" else "none"
    return {
        "case_id": cid,
        "benchmark": rec["benchmark"],
        "prompt": prompt,
        "outcome": outcome,
        "instruction": instr,
        "diagnosed_action": act,
        "gated_by": parsed.get("gated_by"),
        "gated_instruction": parsed.get("gated_instruction"),
        "rechecked_add_subject": parsed.get("rechecked_add_subject"),
        "description": desc,
        "judge_raw": judg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev-bench-dir", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--model", default="glm-5.3-flash")
    args = ap.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["case_id"])
                    except Exception:
                        pass

    recs = []
    for fp in sorted(glob.glob(os.path.join(args.prev_bench_dir, "gen_shards", "*.jsonl"))):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r["case_id"] not in done:
                        recs.append(r)

    print(f"Loaded {len(done)} done, {len(recs)} remaining", flush=True)
    if not recs:
        return

    client = GlmClient(
        api_key=resolve_glm_key(),
        base_url=resolve_glm_endpoint(),
        model=args.model,
        workers=args.workers,
        retries=4,
        timeout=120,
    )

    lock = threading.Lock()
    t0 = time.perf_counter()
    completed = 0
    errors = 0

    def _worker(r):
        for attempt in range(4):
            try:
                return reflect_one(client, r, args.prev_bench_dir)
            except Exception as exc:
                if attempt == 3:
                    return {"case_id": r["case_id"], "benchmark": r["benchmark"], "prompt": r["prompt"],
                            "outcome": "failed_api", "instruction": "", "diagnosed_action": "none", "error": str(exc)}
                time.sleep(1.5 * (attempt + 1))

    with out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_worker, r) for r in recs]
            for fut in as_completed(futs):
                res = fut.result()
                with lock:
                    f_out.write(json.dumps(res, ensure_ascii=False) + "\n")
                    completed += 1
                    if res["outcome"] == "failed_api":
                        errors += 1
                    if completed % 100 == 0 or completed == len(recs):
                        f_out.flush()
                        dt = time.perf_counter() - t0
                        print(f"[{completed}/{len(recs)}] elapsed={dt:.1f}s errors={errors}", flush=True)


if __name__ == "__main__":
    main()
