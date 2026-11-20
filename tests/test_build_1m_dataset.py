"""Offline unit tests for the 1M multi-process + 64-NPU dataset builder."""

import json
from PIL import Image

from scripts.build_1m_dataset_glm import (
    collect_status,
    encode_side_by_side,
    stable_split,
    verify_curate_batch4,
)
from scripts.launch_editor_fleet64 import build_endpoints


def test_build_endpoints_maps_64_chips_across_4_nodes():
    eps = build_endpoints(["n0", "n1", "n2", "n3"], chips_per_node=16, base_port=8100)
    assert len(eps) == 64
    assert eps[0]["url"] == "http://127.0.0.1:8100"
    assert eps[16]["url"] == "http://n1:8100"
    assert eps[63]["url"] == "http://n3:8115"


def test_batch4_verification_and_single_fallback(tmp_path):
    src = tmp_path / "s.png"
    tgt = tmp_path / "t.png"
    Image.new("RGB", (256, 256), (20, 40, 60)).save(src)
    Image.new("RGB", (256, 256), (60, 40, 20)).save(tgt)

    uri = encode_side_by_side(src, tgt, side=128)
    assert uri.startswith("data:image/jpeg;base64,")

    batch = [
        {"uid": f"u{i}", "source_image": str(src), "target_image": str(tgt), "edit_instruction": "Add 1 cup."}
        for i in range(4)
    ]

    class _BatchOkGlm:
        def call(self, messages, **kwargs):
            return {
                "items": [
                    {"idx": i, "task_type": "count_add", "semantic_pass": True, "reason": "ok"}
                    for i in range(4)
                ]
            }

    res = verify_curate_batch4(_BatchOkGlm(), batch)
    assert len(res) == 4 and all(v["semantic_pass"] for _, v, err in res if not err)

    # Fallback when Batch-4 throws an exception
    class _FallbackGlm:
        calls = 0

        def call(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("simulated malformed batch JSON")
            return {"task_type": "object_add", "semantic_pass": True, "reason": "rescued"}

    fb = _FallbackGlm()
    res2 = verify_curate_batch4(fb, batch[:2])
    assert fb.calls == 3  # 1 failed batch + 2 single rescues
    assert len(res2) == 2 and all(v["task_type"] == "object_add" for _, v, _ in res2)


def test_collect_status_counts_shards(tmp_path):
    s0 = tmp_path / "synth64/shards/shard_000"
    c0 = tmp_path / "curate_mp/shards/shard_000"
    s0.mkdir(parents=True)
    c0.mkdir(parents=True)
    (s0 / "manifest.jsonl").write_text('{"uid":"1"}\n{"uid":"2"}\n')
    (c0 / "manifest.jsonl").write_text('{"uid":"3"}\n')
    st = collect_status(tmp_path, target_total=100)
    assert st["verified_total"] == 3
    assert stable_split("l490k:abc") in ("train", "val", "test")
