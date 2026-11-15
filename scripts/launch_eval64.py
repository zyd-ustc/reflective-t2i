"""Launch a multi-rank diagnostic across all four ModelArts workers.

``launch_train64.py`` does this for training; this does the same for the
held-out edit test, so the run uses 64 chips instead of the 16 visible from
worker 0. Each worker starts its own ``torch.distributed.run`` with
``--node_rank`` set, and worker 0 is the rendezvous.

    python scripts/launch_eval64.py --limit 256 --out /data/.../heldout_val256
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
PY = "/data/zyd_workspace/bagel-LatentCoT/venvs/u15-NPU/bin/python"
SOURCE = "/data/zyd_workspace/bagel-LatentCoT/code/sensenova-u15-c35c83b"
MODEL = "/data/zyd_workspace/models/SenseNova-U1.5-8B-MoT"
PER_NODE = 16


def _hosts():
    hosts = [item for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item]
    if len(hosts) != 4:
        raise ValueError("Requires exactly four workers in VC_WORKER_HOSTS")
    return hosts


def _ssh(host, command):
    return ["ssh", "-n", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "root@" + host, "bash", "-lc", shlex.quote(command)]


def _idle(host, index):
    command = (["bash", "-lc", "npu-smi info | grep -c 'No running processes'"]
               if index == 0 else _ssh(host, "npu-smi info | grep -c 'No running processes'"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    text = result.stdout.strip().splitlines()
    return int(text[-1]) if text else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="scripts/heldout_edit_test.py")
    parser.add_argument("--args", default="", help="Extra arguments passed through")
    parser.add_argument("--out", required=True)
    parser.add_argument("--master-port", type=int, default=29516)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    hosts = _hosts()
    output = Path(args.out).resolve()
    log_dir = output.with_name(output.name + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    for index, host in enumerate(hosts):
        idle = _idle(host, index)
        print(f"worker{index} idle chips: {idle}", flush=True)
        if idle != 8:
            raise SystemExit(f"worker{index} is not fully idle ({idle}/8)")

    master = hosts[0]
    inner = (
        f"cd {CODE} && PYTHONPATH={CODE} {PY} -m torch.distributed.run "
        f"--nnodes=4 --nproc_per_node={PER_NODE} --node_rank={{rank}} "
        f"--master_addr={master} --master_port={args.master_port} "
        f"{args.module} {args.args}"
    )
    commands = []
    for index, host in enumerate(hosts):
        body = inner.format(rank=index)
        log = log_dir / f"node{index}.log"
        shell = f"{body} > {log} 2>&1"
        commands.append((host, index, shell))

    if args.dry_run:
        for host, index, shell in commands:
            print(f"[{index}] {host}: {shell}")
        return

    started = []
    for host, index, shell in commands:
        command = (["bash", "-lc", shell] if index == 0 else _ssh(host, shell))
        # Detached: the launcher must not wait for a 40-minute run.
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, start_new_session=True)
        started.append(dict(rank=index, host=host, pid=process.pid))
        time.sleep(3)
    (log_dir / "launch.json").write_text(json.dumps(started, indent=1))
    print(json.dumps(started, indent=1))


if __name__ == "__main__":
    main()
