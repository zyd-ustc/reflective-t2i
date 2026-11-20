"""Manage a 64-chip Qwen-Image-2.1 vLLM-Omni editor fleet across 4 ModelArts nodes (4 x 16 chips)."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_PATH = "/data/zx_wang_shared_work/ModelScope-zx/Qwen-Image-2.1"
SERVED_MODEL = "Qwen/Qwen-Image-2.1"
DEFAULT_FLEET_DIR = "/data/zyd_workspace/isochronic_data_20261107/prod_1m_glm53_20261118/fleet64"
BASE_PORT = 8100


def get_hosts():
    raw = os.environ.get("VC_WORKER_HOSTS", "")
    hosts = [h.strip() for h in raw.split(",") if h.strip()]
    if not hosts:
        raise RuntimeError("VC_WORKER_HOSTS is empty; run this on modelarts-64b")
    return hosts


def build_endpoints(hosts, chips_per_node=16, base_port=BASE_PORT):
    endpoints = []
    idx = 0
    for node_idx, host in enumerate(hosts):
        connect_host = "127.0.0.1" if node_idx == 0 else host
        for chip in range(chips_per_node):
            port = base_port + chip
            endpoints.append(
                dict(
                    id=idx,
                    node=node_idx,
                    chip=chip,
                    host=host,
                    connect_host=connect_host,
                    port=port,
                    url=f"http://{connect_host}:{port}",
                )
            )
            idx += 1
    return endpoints


def run_ssh(host, remote_cmd, timeout=120):
    cmd = [
        "ssh",
        "-n",
        "-p",
        "2222",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{host}",
        remote_cmd,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_endpoint(ep, timeout=3.0):
    try:
        req = urllib.request.Request(ep["url"] + "/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ok = resp.status == 200
    except Exception:
        ok = False
    return ep["id"], ep["node"], ep["chip"], ok


def fleet_status(endpoints):
    with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as pool:
        rows = list(pool.map(check_endpoint, endpoints))
    by_node = {}
    ready = 0
    for _, node, _, ok in rows:
        by_node.setdefault(node, [0, 0])
        by_node[node][1] += 1
        if ok:
            by_node[node][0] += 1
            ready += 1
    return dict(
        ready=ready,
        total=len(endpoints),
        by_node={str(k): f"{v[0]}/{v[1]}" for k, v in sorted(by_node.items())},
        ready_urls=[ep["url"] for ep, (_, _, _, ok) in zip(endpoints, rows) if ok],
    )


def start_local_node(node_idx, fleet_dir, chips_per_node=16, base_port=BASE_PORT):
    fleet = Path(fleet_dir).resolve()
    (fleet / "logs").mkdir(parents=True, exist_ok=True)
    (fleet / "pids").mkdir(parents=True, exist_ok=True)
    if node_idx == 0:
        subprocess.run(["pkill", "-f", "[v]llm serve.*--port 8091"], check=False)
        time.sleep(0.5)

    launched = 0
    skipped = 0
    for chip in range(chips_per_node):
        port = base_port + chip
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1.5) as r:
                if r.status == 200:
                    skipped += 1
                    continue
        except Exception:
            pass

        chip_cmd = (
            "source /data/zyd_workspace/qwen_pe/runtime/Ascend/cann-8.5.1/set_env.sh 2>/dev/null || true; "
            "source /data/zyd_workspace/qwen_pe/runtime/Ascend/nnal/atb/latest/atb/set_env.sh 2>/dev/null || true; "
            "export PYTHONPATH=/data/zyd_workspace/qwen_pe/omni_overlay/codex-omni-bootstrap:"
            "/data/zyd_workspace/qwen_pe/omni_overlay/codex-vllm032-overlay:${PYTHONPATH:-}; "
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn; "
            "export OMP_NUM_THREADS=2; "
            "export TOKENIZERS_PARALLELISM=false; "
            f"export ASCEND_RT_VISIBLE_DEVICES={chip}; "
            f"export MASTER_PORT={29600 + chip * 10}; "
            f"export VLLM_PORT={30600 + chip * 10}; "
            f"export HCCL_IF_BASE_PORT={61000 + chip * 32}; "
            f"exec /data/zyd_workspace/omni-env/bin/python /data/zyd_workspace/omni-env/bin/vllm serve "
            f"{shlex.quote(MODEL_PATH)} --omni --host 0.0.0.0 --port {port} "
            f"--served-model-name {shlex.quote(SERVED_MODEL)} --num-gpus 1 "
            "--dtype bfloat16 --enforce-eager --trust-remote-code"
        )
        log_path = fleet / "logs" / f"node{node_idx}_chip{chip}.log"
        pid_path = fleet / "pids" / f"node{node_idx}_chip{chip}.pid"
        with open(log_path, "ab") as log_fh:
            proc = subprocess.Popen(
                ["bash", "-c", chip_cmd],
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        pid_path.write_text(str(proc.pid) + "\n")
        launched += 1
        time.sleep(0.35)

    return dict(node=node_idx, launched=launched, skipped=skipped)


def start_fleet(fleet_dir, chips_per_node=16, base_port=BASE_PORT):
    hosts = get_hosts()
    fleet = Path(fleet_dir).resolve()
    (fleet / "logs").mkdir(parents=True, exist_ok=True)
    (fleet / "pids").mkdir(parents=True, exist_ok=True)
    endpoints = build_endpoints(hosts, chips_per_node=chips_per_node, base_port=base_port)
    (fleet / "endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")

    py = sys.executable
    script = str(Path(__file__).resolve())
    results = [start_local_node(0, str(fleet), chips_per_node, base_port)]
    for node_idx, host in list(enumerate(hosts))[1:]:
        remote_cmd = (
            f"{shlex.quote(py)} {shlex.quote(script)} --action start-local "
            f"--node-idx {node_idx} --fleet-dir {shlex.quote(str(fleet))} "
            f"--chips-per-node {chips_per_node} --base-port {base_port}"
        )
        res = run_ssh(host, remote_cmd, timeout=120)
        results.append(dict(node=node_idx, host=host, rc=res.returncode, out=res.stdout.strip(), err=res.stderr.strip()[-200:]))
    return dict(fleet_dir=str(fleet), endpoints_file=str(fleet / "endpoints.json"), nodes=results)


def stop_fleet(fleet_dir):
    hosts = get_hosts()
    out = []
    subprocess.run(["pkill", "-f", "[v]llm serve.*Qwen-Image-2.1"], check=False)
    out.append(dict(node=0, rc=0))
    for node_idx, host in list(enumerate(hosts))[1:]:
        res = run_ssh(host, 'pkill -f "[v]llm serve.*Qwen-Image-2.1" 2>/dev/null || true; echo stopped')
        out.append(dict(node=node_idx, rc=res.returncode, out=res.stdout.strip()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("start", "start-local", "status", "wait", "stop"), required=True)
    p.add_argument("--node-idx", type=int, default=0)
    p.add_argument("--fleet-dir", default=DEFAULT_FLEET_DIR)
    p.add_argument("--chips-per-node", type=int, default=16)
    p.add_argument("--base-port", type=int, default=BASE_PORT)
    p.add_argument("--wait-timeout", type=int, default=420)
    args = p.parse_args()

    if args.action == "start-local":
        print(json.dumps(start_local_node(args.node_idx, args.fleet_dir, args.chips_per_node, args.base_port)))
        return

    hosts = get_hosts()
    endpoints = build_endpoints(hosts, chips_per_node=args.chips_per_node, base_port=args.base_port)

    if args.action == "start":
        print(json.dumps(start_fleet(args.fleet_dir, args.chips_per_node, args.base_port), indent=2))
    elif args.action == "status":
        print(json.dumps(fleet_status(endpoints), indent=2))
    elif args.action == "wait":
        deadline = time.time() + args.wait_timeout
        while time.time() < deadline:
            st = fleet_status(endpoints)
            print(json.dumps({k: st[k] for k in ("ready", "total", "by_node")}), flush=True)
            if st["ready"] == st["total"]:
                return
            time.sleep(8)
        st = fleet_status(endpoints)
        print(json.dumps(st, indent=2))
        if st["ready"] == 0:
            raise SystemExit("No editor endpoints became ready within timeout")
    elif args.action == "stop":
        print(json.dumps(stop_fleet(args.fleet_dir), indent=2))


if __name__ == "__main__":
    main()
