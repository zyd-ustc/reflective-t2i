"""Report training progress, health, and ETA for a running stage.

Exits nonzero on a stalled or crashed job so a scheduler can react.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def npu_used():
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return None
    return out.count("No running processes found in NPU")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--expected-budget", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out = args.output
    rows = read_rows(out / "train.jsonl")
    state = dict(output=str(out), steps=len(rows), status="unknown")

    if not rows:
        state["status"] = "waiting"
        print(json.dumps(state) if args.json else "waiting: no steps yet")
        return 0

    last = rows[-1]
    state["last_step"] = last["step"]
    state["loss"] = round(float(last["loss"]), 5)
    state["lr"] = float(last["lr"])
    state["mode"] = last.get("mode")
    state["keep"] = last.get("keep")
    state["prompt_dropped"] = last.get("prompt_dropped")
    state["peak_memory_gib"] = round(float(last.get("peak_memory_gib") or 0), 1)

    tail = [float(r["seconds"]) for r in rows[-200:]]
    per_step = sum(tail) / len(tail)
    state["sec_per_step"] = round(per_step, 2)

    first = next((r for r in rows if r.get("total_steps")), None)
    total = args.total_steps or (int(first["total_steps"]) if first else 0)
    if total:
        remaining = max(0, total - last["step"])
        state["total_steps"] = total
        state["progress"] = round(last["step"] / total, 4)
        state["eta_hours"] = round(remaining * per_step / 3600, 2)
        state["eta_finish"] = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(time.time() + remaining * per_step)
        )

    age = time.time() - (out / "train.jsonl").stat().st_mtime
    state["seconds_since_last_row"] = int(age)
    used = npu_used()
    if used is not None:
        state["npu_busy_chips"] = 8 - used if 0 <= used <= 8 else None

    if age > args.stall_minutes * 60:
        state["status"] = "stalled"
    elif total and last["step"] >= total:
        state["status"] = "finished"
    elif state["status"] != "crashed":
        state["status"] = "running"

    # A budget that disagrees with the data size stops training early; surface it.
    ckpts = sorted(p for p in out.glob("step_*") if p.is_dir())
    if ckpts:
        manifest_path = ckpts[-1] / "manifest.json"
        if manifest_path.exists():
            try:
                stored = json.loads(manifest_path.read_text())
                expected = args.expected_budget or 0
                if expected and int(stored.get("total_steps", 0)) not in (expected,):
                    state["budget_mismatch"] = dict(
                        stored=stored.get("total_steps"), expected=expected
                    )
                    state["status"] = "budget_mismatch"
            except (json.JSONDecodeError, OSError):
                pass
    state["checkpoints"] = [p.name for p in ckpts[-3:]]
    # Only tracebacks newer than the last logged step mean this run is broken;
    # an old crash stays in the log after a successful resume.
    log = out.with_name(out.name + "_logs") / "node0.log"
    state["fresh_tracebacks"] = 0
    if log.exists():
        cutoff = (out / "train.jsonl").stat().st_mtime if (out / "train.jsonl").exists() else 0
        state["fresh_tracebacks"] = sum(
            1 for idx, line in enumerate(log.read_text(errors="ignore").splitlines())
            if "Traceback" in line and log.stat().st_mtime > cutoff
        )
    if state["fresh_tracebacks"]:
        state["status"] = "crashed"

    if args.json:
        print(json.dumps(state))
    else:
        fields = dict(state)
        fields["progress"] = state.get("progress", 0.0) * 100
        fields["total_steps"] = state.get("total_steps", "?")
        fields["eta_hours"] = state.get("eta_hours", "?")
        fields["eta_finish"] = state.get("eta_finish", "?")
        print(
            "step {last_step}/{total_steps} ({progress:.1f}%)  loss {loss}  "
            "lr {lr:.3e}  {sec_per_step}s/step  ETA {eta_hours}h -> {eta_finish}  "
            "mem {peak_memory_gib}GiB  [{status}]".format(**fields)
        )
        if state.get("checkpoints"):
            print("  recent checkpoints:", ", ".join(state["checkpoints"]))
    return 1 if state["status"] in ("stalled", "crashed", "budget_mismatch") else 0


if __name__ == "__main__":
    raise SystemExit(main())
