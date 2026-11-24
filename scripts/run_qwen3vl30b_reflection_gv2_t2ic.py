#!/usr/bin/env python3
"""Run local Qwen3-VL-30B-A3B-Instruct reflection (DESCRIBE -> JUDGE + Breakthrough-A gates) across 32x 2-NPU workers."""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("REFLECTIVE_WORKSPACE", ROOT.parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

DEFAULT_MODEL = "/data/ltj/modelscope/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_QWEN_PY = str(WORKSPACE / "envs/qwen3vl-score/bin/python")
HOSTS_FILE = Path(os.environ.get("HOSTS_FILE", "/root/hosts"))


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


def cmd_worker(args):
    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out_dir = Path(args.out).resolve()
    shard_dir = out_dir / "reflect_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"reflect_rank{args.rank:02d}.jsonl"
    status_path = shard_dir / f"done_rank{args.rank:02d}.json"

    done = set()
    if shard_path.exists():
        for line in shard_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["case_id"])

    all_recs = []
    for fp in sorted(glob.glob(os.path.join(args.prev_bench_dir, "gen_shards", "*.jsonl"))):
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if line.strip():
                all_recs.append(json.loads(line))
    all_recs.sort(key=lambda r: r["case_id"])
    mine = [r for i, r in enumerate(all_recs) if i % args.world == args.rank]

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    ).eval()

    def _ask(sys_msg, user_text, pil_img=None, max_new_tokens=384):
        content = []
        if pil_img is not None:
            content.append({"type": "image", "image": pil_img})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
            {"role": "user", "content": content},
        ]
        prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if pil_img is not None:
            inputs = proc(text=[prompt_text], images=[pil_img], return_tensors="pt").to("npu:0")
        else:
            inputs = proc(text=[prompt_text], return_tensors="pt").to("npu:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        return raw.rsplit("</think>", 1)[-1].strip()

    for pos, rec in enumerate(mine, 1):
        cid = rec["case_id"]
        if cid in done:
            continue
        t0 = time.perf_counter()
        prompt = rec["prompt"]
        prev_path = Path(args.prev_bench_dir) / Path(rec["arms"]["baseline"]["image_rel"]).parent / "preview.png"
        with Image.open(prev_path) as im:
            pil_img = im.convert("RGB")

        desc = _ask(DESCRIBE_SYSTEM, DESCRIBE_TEMPLATE, pil_img=pil_img, max_new_tokens=384)
        judg = _ask(JUDGE_SYSTEM, JUDGE_TEMPLATE.format(prompt=prompt, description=desc), pil_img=None, max_new_tokens=320)
        parsed = parse_reflection(judg, desc)
        initial_instr = parsed["instruction"]

        if parsed["outcome"] == "edit":
            subj = extract_add_subject(parsed["instruction"])
            if subj:
                fscan = _ask(DESCRIBE_SYSTEM, FOCUSED_SCAN_TEMPLATE.format(subject=subj), pil_img=pil_img, max_new_tokens=192)
                comb = desc + f"\n\nFocused re-scan for '{subj}':\n" + fscan
                judg2 = _ask(JUDGE_SYSTEM, JUDGE_TEMPLATE.format(prompt=prompt, description=comb), pil_img=None, max_new_tokens=320)
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
        out_rec = {
            "case_id": cid,
            "benchmark": rec["benchmark"],
            "prompt": prompt,
            "outcome": outcome,
            "instruction": instr,
            "diagnosed_action": act,
            "gated_by": parsed.get("gated_by"),
            "gated_instruction": parsed.get("gated_instruction"),
            "rechecked_add_subject": parsed.get("rechecked_add_subject"),
            "description": desc[:600],
            "judge_raw": judg[:400],
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "rank": args.rank,
        }
        with shard_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        status_path.write_text(json.dumps({"rank": args.rank, "done": pos, "total": len(mine), "finished": False}) + "\n")
        print(f"[reflect30b {args.rank:02d}] ({pos}/{len(mine)}) {cid} | outcome={outcome} ({act}) gated={parsed.get('gated_by')}", flush=True)

    status_path.write_text(json.dumps({"rank": args.rank, "done": len(mine), "total": len(mine), "finished": True}) + "\n")


def cmd_launch32(args):
    import shlex
    hosts = [item.strip() for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item.strip()]
    if len(hosts) != 4:
        raise ValueError(f"Expected 4 hosts in VC_WORKER_HOSTS, got {len(hosts)}")
    out_dir = Path(args.out).resolve()
    log_dir = out_dir / "reflect_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    world = len(hosts) * 4
    helper_path = ROOT / "scripts" / "start_train_node.py"
    helper_py = sys.executable

    for node_idx, host in enumerate(hosts):
        lines = [
            "set -euo pipefail",
            "set +u; source /usr/local/Ascend/ascend-toolkit/set_env.sh; set -u",
            f"mkdir -p {shlex.quote(str(log_dir))}",
            f"cd {shlex.quote(str(ROOT))}",
            f"export PYTHONPATH={shlex.quote(str(ROOT))}:${{PYTHONPATH:-}}",
            "export PYTHONUNBUFFERED=1",
            "export OMP_NUM_THREADS=1",
            "export TOKENIZERS_PARALLELISM=false",
            "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True",
        ]
        for group_idx in range(4):
            gr = node_idx * 4 + group_idx
            devs = ",".join(str(group_idx * 4 + k) for k in range(4))
            rk_log = log_dir / f"reflect_rank{gr:02d}.log"
            rk_pid = log_dir / f"reflect_rank{gr:02d}.pid"
            cmd = [
                helper_py,
                str(helper_path),
                "--log", str(rk_log),
                "--pid-file", str(rk_pid),
                "--",
                args.python,
                str(ROOT / "scripts" / "run_qwen3vl30b_reflection_gv2_t2ic.py"),
                "worker",
                "--model", args.model,
                "--prev-bench-dir", args.prev_bench_dir,
                "--out", str(out_dir),
                "--rank", str(gr),
                "--world", str(world),
            ]
            lines.append(f"ASCEND_RT_VISIBLE_DEVICES={devs} " + shlex.join(cmd))
            if group_idx + 1 < 4:
                lines.append("sleep 0.5")
        lines.append("echo NODE_LAUNCHED_4")
        shell = "\n".join(lines)
        if node_idx == 0:
            res = subprocess.run(["bash", "-lc", shell], capture_output=True, text=True, timeout=60)
        else:
            res = subprocess.run(
                [
                    "ssh", "-n", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                    host, "bash", "-lc", shell,
                ],
                capture_output=True, text=True, timeout=60,
            )
        if res.returncode != 0 or "NODE_LAUNCHED_4" not in res.stdout:
            raise RuntimeError(f"Failed launching node {node_idx}: {res.stdout}\n{res.stderr}")
    print(f"Launched {world} 4-NPU Qwen3-VL-30B-A3B reflection workers -> {out_dir}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("worker")
    pw.add_argument("--model", default=DEFAULT_MODEL)
    pw.add_argument("--prev-bench-dir", required=True)
    pw.add_argument("--out", required=True)
    pw.add_argument("--devices", default="0,1")
    pw.add_argument("--rank", type=int, required=True)
    pw.add_argument("--world", type=int, required=True)

    pl = sub.add_parser("launch32")
    pl.add_argument("--model", default=DEFAULT_MODEL)
    pl.add_argument("--python", default=DEFAULT_QWEN_PY)
    pl.add_argument("--prev-bench-dir", required=True)
    pl.add_argument("--out", required=True)

    args = p.parse_args()
    {"worker": cmd_worker, "launch32": cmd_launch32}[args.cmd](args)


if __name__ == "__main__":
    main()
