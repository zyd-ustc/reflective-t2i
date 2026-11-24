"""Launch reflective_t2i_v1 large-scale 64-NPU DDP training on four 16-chip ModelArts workers (modelarts-64b)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

CODE = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("REFLECTIVE_WORKSPACE", CODE.parents[1]))
DEFAULT_PY = str(WORKSPACE / "venvs/u15-NPU/bin/python")
DEFAULT_ZIMAGE_CORE = "/data/zyd_workspace/zimage-isochronic-correction/src"
DEFAULT_SD35_DEPS = str(WORKSPACE / "venvs/sd35-deps")


def _hosts() -> list[str]:
    hosts = [item.strip() for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item.strip()]
    if len(hosts) != 4:
        raise ValueError(f"Expected 4 hosts in VC_WORKER_HOSTS for modelarts-64b, got {len(hosts)}")
    return hosts


def _remote(host: str, index: int, command: str) -> list[str]:
    if index == 0:
        return ["bash", "-lc", command]
    return [
        "ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "root@" + host, "bash", "-lc", shlex.quote(command),
    ]


def _preflight(hosts: list[str], output: Path, *, resuming: bool) -> None:
    if output.exists() and not resuming:
        raise ValueError(f"Output directory already exists (pass --resume or pick a fresh --output): {output}")
    for idx, host in enumerate(hosts):
        res = subprocess.run(
            _remote(host, idx, "npu-smi info | grep -c 'No running processes found in NPU '"),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if res.returncode != 0:
            raise RuntimeError(f"NPU preflight failed on {host}: {res.stderr}")
        idle = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "0"
        if idle != "8":
            raise RuntimeError(f"Host {host} not fully idle ({idle}/8 NPU packages)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Path to reflective_t2i_v1 64-NPU JSON config")
    p.add_argument("--output", required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--manifest", default=None, help="Optional manifest glob override")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--master-port", type=int, default=30051)
    p.add_argument("--code", default=str(CODE))
    p.add_argument("--python", default=DEFAULT_PY)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-preflight", action="store_true")
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg.get("protocol") != "reflective_t2i_v1":
        raise ValueError(f"Expected reflective_t2i_v1 config, got {cfg.get('protocol')}")

    vc_hosts = [item.strip() for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item.strip()]
    hosts = vc_hosts if (not args.dry_run or len(vc_hosts) == 4) else ["node0", "node1", "node2", "node3"]
    if len(hosts) != 4:
        raise ValueError(f"Expected 4 hosts in VC_WORKER_HOSTS for modelarts-64b, got {len(hosts)}")
    output = Path(args.output).resolve()
    log_dir = output.with_name(output.name + "_logs")
    code = Path(args.code).resolve()
    helper = code / "scripts/start_train_node.py"

    max_steps = args.max_steps or int(cfg["training"].get("max_steps", 12000))
    save_every = int(cfg["training"].get("save_every", 1000))
    grad_accum = int(cfg["training"].get("grad_accum", 1))

    py_path = f"{code}:{cfg['generator'].get('zimage_core_src') or DEFAULT_ZIMAGE_CORE}:{DEFAULT_SD35_DEPS}"
    launch_env = {
        "PYTHONPATH": py_path,
        "PYTHONUNBUFFERED": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(i) for i in range(16)),
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HCCL_IF_BASE_PORT": "64720",
        "HCCL_CONNECT_TIMEOUT": "3600",
        "HCCL_EXEC_TIMEOUT": "600",
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }
    env_text = " ".join(f"{k}={shlex.quote(v)}" for k, v in launch_env.items())

    launches = []
    for idx, host in enumerate(hosts):
        cmd = [
            args.python, "-m", "torch.distributed.run",
            "--nnodes=4", "--nproc_per_node=16",
            f"--node_rank={idx}", f"--master_addr={hosts[0]}",
            f"--master_port={args.master_port}",
            str(code / "scripts/run_reflective_t2i.py"), "train",
            "--config", str(cfg_path),
            "--out", str(output),
            "--max-steps", str(max_steps),
            "--save-every", str(save_every),
            "--grad-accum", str(grad_accum),
            "--device", "npu:0",
        ]
        if args.manifest:
            cmd += ["--manifest", args.manifest]
        if args.resume:
            cmd += ["--resume", str(Path(args.resume).resolve())]

        log_file = log_dir / f"node{idx}.log"
        pid_file = log_dir / f"node{idx}.pid"
        helper_cmd = [args.python, str(helper), "--log", str(log_file), "--pid-file", str(pid_file), "--", *cmd]
        shell = f"cd {shlex.quote(str(code))}; env {env_text} {shlex.join(helper_cmd)}"
        launches.append({"node": idx, "host": host, "log": str(log_file), "pid_file": str(pid_file), "shell": shell})

    print(json.dumps({"output": str(output), "log_dir": str(log_dir), "launches": launches}, indent=2), flush=True)
    if args.dry_run:
        return
    if not args.skip_preflight:
        _preflight(hosts, output, resuming=bool(args.resume))

    log_dir.mkdir(parents=True, exist_ok=True)
    for item in launches:
        res = subprocess.run(_remote(item["host"], item["node"], item["shell"]), capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to launch node {item['node']} ({item['host']}): {res.stderr}")
        time.sleep(0.5)
    print(f"Launched 64-NPU reflective_t2i_v1 training -> {output} (logs: {log_dir})", flush=True)


if __name__ == "__main__":
    main()
