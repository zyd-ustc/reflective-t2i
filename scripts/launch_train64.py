"""Launch unified isochronic manifold training on four 16-chip ModelArts workers."""

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


def _hosts():
    hosts = [item for item in os.environ.get("VC_WORKER_HOSTS", "").split(",") if item]
    if len(hosts) != 4:
        raise ValueError("Requires exactly four workers in VC_WORKER_HOSTS")
    return hosts


def _remote(host, index, command):
    if index == 0:
        return ["bash", "-lc", command]
    return [
        "ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "root@" + host, "bash", "-lc", shlex.quote(command),
    ]


def _ssh_command(host, command):
    """SSH command used as a detached local process, never awaited."""
    return [
        "ssh", "-n", "-p", "2222", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "root@" + host, "bash", "-lc", shlex.quote(command),
    ]


def _preflight(hosts, output, resuming=False):
    if output.exists() and not resuming:
        raise ValueError("Output directory already exists: " + str(output))
    for index, host in enumerate(hosts):
        command = _remote(
            host,
            index,
            "npu-smi info | grep -c 'No running processes found in NPU '",
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("NPU preflight failed on " + host + "\n" + result.stdout + result.stderr)
        idle = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "0"
        if str(idle) != "8":
            raise RuntimeError("NPU not fully idle on " + host + ": idle=" + str(idle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", nargs="+", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--master-port", type=int, default=30041)
    parser.add_argument("--config", help="Configuration; defaults to isochronic_manifold.json")
    parser.add_argument("--code", default=str(CODE))
    parser.add_argument("--python", default=PY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    hosts = _hosts()
    output = Path(args.output).resolve()
    log_dir = output.with_name(output.name + "_logs")
    default_cfg = "isochronic_manifold.json"
    from qwen_latent_cot.u15_target_correction.config import load_config
    from qwen_latent_cot.u15_target_correction.checkpoint import geometry_gate

    stage_cfg = (
        Path(args.config).resolve()
        if args.config
        else Path(args.code).resolve() / "configs/u15_isochronic_state" / default_cfg
    )
    load_config(stage_cfg)

    def _gate(path, label):
        manifest = Path(path).resolve() / "manifest.json"
        if not manifest.exists():
            parser.error(f"{label} has no manifest.json: {path}")
        stored = json.loads(manifest.read_text()).get("identity", {})
        try:
            geometry_gate(stored)
        except ValueError as error:
            parser.error(f"{label}: {error}")

    for path, label in ((getattr(args, "resume", None), "--resume"), (args.init_checkpoint, "--init-checkpoint")):
        if path:
            _gate(path, label)

    launch_env = dict(
        PYTHONPATH=str(Path(args.code).resolve()),
        PYTHONUNBUFFERED="1",
        ASCEND_RT_VISIBLE_DEVICES=",".join(str(i) for i in range(16)),
        HCCL_NPU_SOCKET_PORT_RANGE="auto",
        HCCL_IF_BASE_PORT="64720",
        HCCL_CONNECT_TIMEOUT="3600",
        HCCL_EXEC_TIMEOUT="600",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_NPU_ALLOC_CONF="expandable_segments:True",
    )
    for key in (
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE", "WANDB_DIR",
        "WANDB_RUN_NAME", "WANDB_TAGS", "WANDB_BASE_URL",
        "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
    ):
        if key in os.environ:
            launch_env[key] = os.environ[key]
    env_text = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in launch_env.items())

    code = Path(args.code)
    helper = code / "scripts/start_train_node.py"
    launches = []
    for index, host in enumerate(hosts):
        command = [
            args.python, "-m", "torch.distributed.run",
            "--nnodes=4", "--nproc_per_node=16",
            f"--node_rank={index}", f"--master_addr={hosts[0]}",
            f"--master_port={args.master_port}",
            "-m", "qwen_latent_cot.u15_target_correction.cli", "train",
            "--distributed", "--device", "npu:0",
            "--model", MODEL, "--source", SOURCE,
            "--config", str(stage_cfg),
            "--output", str(output),
            "--linear", *args.linear,
        ]
        if args.resume:
            command += ["--resume", args.resume]
        if args.init_checkpoint:
            command += ["--init-checkpoint", args.init_checkpoint]
        if args.steps:
            command += ["--steps", str(args.steps)]
        log = log_dir / f"node{index}.log"
        pid_file = log_dir / f"node{index}.pid"
        helper_command = [
            args.python, str(helper), "--log", str(log), "--pid-file", str(pid_file), "--", *command,
        ]
        shell = f"cd {shlex.quote(str(code))}; env {env_text} {shlex.join(helper_command)}"
        launches.append(
            dict(node=index, host=host, log=str(log), pid_file=str(pid_file), command=helper_command, shell=shell)
        )

    print(json.dumps(dict(output=str(output), log_dir=str(log_dir), launches=launches), indent=2))
    if args.dry_run:
        return
    if not args.skip_preflight:
        _preflight(hosts, output, resuming=bool(args.resume))
    log_dir.mkdir(parents=True, exist_ok=True)
    pids = []
    for item in launches:
        Path(item["pid_file"]).unlink(missing_ok=True)
        command = ["bash", "-lc", item["shell"]] if item["node"] == 0 else _ssh_command(item["host"], item["shell"])
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_path = Path(item["pid_file"])
        deadline = time.monotonic() + 90
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        if not pid_path.exists():
            raise RuntimeError(f"Node {item['node']} did not write a PID file")
        pids.append(dict(node=item["node"], host=item["host"], pid=pid_path.read_text().strip(), log=item["log"]))
    (log_dir / "pids.json").write_text(json.dumps(pids, indent=2) + "\n")
    print(json.dumps(dict(launched=pids), indent=2))


if __name__ == "__main__":
    main()
