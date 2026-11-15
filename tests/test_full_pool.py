"""Full-pool traversal: record-equal shuffled epochs, family splits, and budget by data."""

import json

import pytest

from qwen_latent_cot.u15_target_correction.config import stage_updates


def _cfg(**overrides):
    cfg = dict(
        global_batch=4,
        epochs=1,
        seed=3,
        keep_fraction=0.0,
        lora_rank=4,
        read_time=0.36,
        state_times=[0.36],
        state_time_probs=[1.0],
        keep_instruction="keep unchanged",
    )
    cfg.update(overrides)
    return cfg


class _Pool:
    """A minimal manifest stand-in exposing exactly what TrainingSampler touches."""

    def __init__(self, n):
        self.train_indices = list(range(n))
        self.rows = [dict(i=i) for i in range(n)]
        self.cfg = None

    @property
    def count(self):
        return len(self.train_indices)

    def choose_time(self, rng, time=None):
        return 0.36

    def sample(self, index, time, keep=False):
        return dict(uid="u%d" % index, keep=bool(keep), time=time, role="L_edit")

    def role_pools(self):
        return {}


def test_every_record_is_drawn_once_per_epoch():
    from qwen_latent_cot.u15_target_correction.data import TrainingSampler

    sampler = TrainingSampler(_Pool(10), _cfg())
    drawn = [sampler.draw()["uid"] for _ in range(10)]
    assert sorted(drawn) == sorted("u%d" % i for i in range(10))
    # The 11th draw opens the next epoch rather than repeating the same order.
    assert sampler.draw()["epoch"] == 1


def test_consecutive_epochs_shuffle_differently():
    from qwen_latent_cot.u15_target_correction.data import TrainingSampler

    sampler = TrainingSampler(_Pool(30), _cfg())
    first = [sampler.draw()["uid"] for _ in range(30)]
    second = [sampler.draw()["uid"] for _ in range(30)]
    assert sorted(first) == sorted(second)
    assert first != second


def test_task_weights_is_rejected():
    from qwen_latent_cot.u15_target_correction.data import TrainingSampler

    with pytest.raises(ValueError, match="task_weights is retired"):
        TrainingSampler(_Pool(10), _cfg(task_weights={"T8": 0.3}))


def test_resume_continues_inside_the_epoch():
    from qwen_latent_cot.u15_target_correction.data import TrainingSampler

    walk = TrainingSampler(_Pool(20), _cfg())
    for _ in range(7):
        walk.draw()
    state = walk.state_dict()
    assert state["cursor"] == 7

    resumed = TrainingSampler(_Pool(20), _cfg())
    resumed.load_state_dict(state)
    assert resumed.cursor == 7
    # The remaining draws must not repeat anything already consumed.
    assert walk.draw()["uid"] == resumed.draw()["uid"]
    assert walk.draw()["uid"] == resumed.draw()["uid"]


def test_budget_counts_epochs_over_the_whole_set():
    cfg = _cfg(global_batch=64, epochs=2)
    one_epoch = 15_109_804 / 64
    assert stage_updates(15_109_804, cfg, global_batch=64) == 2 * int(-(-15_109_804 // 64))
    assert stage_updates(15_109_804, cfg, global_batch=64) > 470_000
    assert one_epoch > 236_000


def test_budget_is_not_capped_by_a_legacy_max_step():
    cfg = _cfg(global_batch=64, epochs=1, max_steps=14_723)
    assert stage_updates(15_109_804, cfg, global_batch=64) == 236_091


def test_epochs_must_be_positive():
    with pytest.raises(ValueError, match="epochs"):
        stage_updates(100, _cfg(epochs=0), global_batch=4)


def test_shipped_configs_drop_task_weights_and_use_epochs():
    from pathlib import Path

    from qwen_latent_cot.u15_target_correction.config import load_config

    directory = Path(__file__).resolve().parent.parent / "configs/u15_isochronic_state"
    for name in ("stage1_full490k.json", "editor_v2_rollout_lora.json",
                 "editor_v2_rollout_gen_full.json", "tail_v1_lora.json"):
        cfg = load_config(directory / name)
        assert "task_weights" not in cfg, name
        assert "max_steps" not in cfg, name
        assert int(cfg["epochs"]) >= 1, name


def test_family_split_never_crosses_a_boundary():
    from scripts.build_unimoe_edit_manifest import split_for_family

    splits = {split_for_family("%08x" % (i * 7_919_001 % 0xFFFFFFFF), 0.01, 0.01) for i in range(400)}
    assert splits == {"train", "val", "test"}
    # Deterministic for the same family, so a rerun reproduces the split.
    assert split_for_family("deadbeef", 0.01, 0.01) == split_for_family("deadbeef", 0.01, 0.01)


def test_source_family_groups_by_image_not_by_record():
    from scripts.build_unimoe_edit_manifest import source_family

    a = source_family("ImgEdit", "/data/x/photo.jpg")
    b = source_family("ImgEdit", "/data/x/photo.jpg")
    c = source_family("ImgEdit", "/data/x/other.jpg")
    d = source_family("MagicBrush", "/data/x/photo.jpg")
    assert a == b and a != c
    # The same picture from two collections stays in two families.
    assert a != d


def test_instruction_wrapper_is_stripped_but_the_action_kept():
    from scripts.build_unimoe_edit_manifest import normalize_instruction

    assert normalize_instruction("<image>\nEditing task: Put a backpack on the bed.") == "Put a backpack on the bed."
    assert normalize_instruction("Image modification: The person lowers the record.") == "The person lowers the record."
    assert normalize_instruction("remove the sandwich and add a burger") == "remove the sandwich and add a burger"
    # A real action verb is never treated as a wrapper.
    assert normalize_instruction("Edit the sky to night") == "Edit the sky to night"


def test_manifest_shards_cover_the_global_order_exactly_once(tmp_path):
    """Each rank reads only its stride, and together they cover every record."""
    import json as _json

    from PIL import Image

    from qwen_latent_cot.u15_target_correction.data import LinearManifest

    class _RT:
        width = height = 16
        device = "cpu"
        identity = {"stub": True}

        def noise_scale(self):
            return 1.0

        def times(self):
            return [0.0, 0.36, 1.0]

    cfg = _cfg()
    cfg.update(
        condition_protocol="isochronic_condition_v2", width=16, height=16, num_steps=4,
        edit_steps=4, lookahead_steps=2, timestep_shift=3.0, cfg_scale=4.0,
        image_cfg_scale=1.0, cfg_norm="none", cfg_interval=[0.0, 1.0], t_eps=0.05,
        inversion={},
    )
    image = tmp_path / "i.png"
    Image.new("RGB", (16, 16)).save(image)
    rows = [
        dict(uid="u%d" % i, split="train", source_image=str(image), target_image=str(image),
             edit_instruction="add one cat", tail_condition="a cat",
             source_family="f%d" % i, prompt_family="f%d" % i)
        for i in range(12)
    ]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("".join(_json.dumps(r) + "\n" for r in rows))

    seen = []
    for rank in range(4):
        shard = LinearManifest(manifest, _RT(), cfg, rank=rank, world=4)
        assert shard.count == 3
        seen.extend(row["global_index"] for row in shard.rows)
    assert sorted(seen) == list(range(12))


def test_prune_keeps_the_newest_checkpoints(tmp_path):
    """The shared volume filled up mid-run, so the window is bounded."""
    from unittest import mock

    from qwen_latent_cot.u15_target_correction import checkpoint

    for step in (2000, 4000, 6000, 8000):
        (tmp_path / f"step_{step:06d}").mkdir()
    with mock.patch.object(checkpoint.dist, "rank", return_value=0):
        removed = checkpoint.prune_checkpoints(tmp_path, 2)
    assert sorted(removed) == ["step_002000", "step_004000"]
    assert sorted(p.name for p in tmp_path.glob("step_*")) == ["step_006000", "step_008000"]
    # Zero disables pruning entirely.
    with mock.patch.object(checkpoint.dist, "rank", return_value=0):
        assert checkpoint.prune_checkpoints(tmp_path, 0) == []


def test_missing_checkpoint_directory_reports_storage_not_serialisation(tmp_path):
    from qwen_latent_cot.u15_target_correction.checkpoint import _assert_writable

    with pytest.raises(RuntimeError, match="never became visible"):
        _assert_writable(tmp_path / "never_created")


def test_budget_is_derived_from_all_ranks_not_one_shard(tmp_path):
    """Passing world * count once under-reported the budget by a factor of world."""
    from qwen_latent_cot.u15_target_correction.train import _global_train_records

    class _Manifest:
        count = 232574  # this rank's shard

    shard_dir = tmp_path / "shards64"
    shard_dir.mkdir()
    per_rank = 14882904 // 64
    index = {str(i): {"rows": per_rank, "splits": {"train": per_rank}} for i in range(64)}
    index["63"]["splits"]["train"] += 14882904 - per_rank * 64
    (shard_dir / "shards.json").write_text(json.dumps(index))

    manifest_path = tmp_path / "L.jsonl"
    manifest_path.write_text("")
    total = _global_train_records(str(manifest_path), 64, True, _Manifest())
    assert total == 14882904
    assert stage_updates(total, _cfg(global_batch=64, epochs=2), global_batch=64) == 465092


def test_single_shard_count_is_not_used_as_the_budget_denominator(tmp_path):
    """The old expression produced 7,268 steps from a 232k shard; it must not."""
    from qwen_latent_cot.u15_target_correction.train import _global_train_records

    class _Manifest:
        count = 232574

    shard_dir = tmp_path / "shards64"
    shard_dir.mkdir()
    per_rank = 14882904 // 64
    index = {str(i): {"splits": {"train": per_rank}} for i in range(64)}
    (shard_dir / "shards.json").write_text(json.dumps(index))
    path = tmp_path / "L.jsonl"
    path.write_text("")

    total = _global_train_records(str(path), 64, True, _Manifest())
    budget = stage_updates(total, _cfg(global_batch=64, epochs=2), global_batch=64)
    assert budget > 400_000
    assert budget != 7268


def test_tail_manifest_and_sampler_shards_cover_global_order_once(tmp_path):
    """Each rank reads only its tail stride and draws distinct records across ranks."""
    from PIL import Image

    from qwen_latent_cot.u15_target_correction.data import TailManifest, TailSampler

    class _RT:
        width = height = 16
        device = "cpu"

    image = tmp_path / "t.png"
    Image.new("RGB", (16, 16)).save(image)
    rows = [
        dict(uid=f"t{i}", split="train", target_image=str(image), tail_condition=f"cat {i}")
        for i in range(12)
    ]
    manifest_path = tmp_path / "tail.jsonl"
    manifest_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    cfg = _cfg()
    seen_indices = []
    drawn_uids = []
    for rank in range(4):
        shard = TailManifest(manifest_path, _RT(), cfg, rank=rank, world=4)
        assert shard.count == 3
        seen_indices.extend(row["global_index"] for row in shard.rows)
        sampler = TailSampler(shard, cfg, rank=rank, world=4)
        drawn_uids.extend(sampler.draw()["uid"] for _ in range(3))
    assert sorted(seen_indices) == list(range(12))
    assert sorted(drawn_uids) == sorted(f"t{i}" for i in range(12))


def test_global_train_records_without_pre_split_sums_stride_ranks(tmp_path):
    """Stride-filtered unsharded runs sum counts across ranks instead of using 1/world."""
    from types import SimpleNamespace
    from unittest import mock

    from qwen_latent_cot.u15_target_correction import train as train_mod

    path = tmp_path / "m.jsonl"
    path.write_text("")
    manifest = SimpleNamespace(count=25, rank=0, rt=SimpleNamespace(device="cpu"))
    with mock.patch.object(train_mod.dist, "world_size", return_value=4), \
         mock.patch.object(train_mod.dist, "sum_int", return_value=101) as mocked_sum:
        total = train_mod._global_train_records(str(path), 4, False, manifest)
    assert total == 101
    mocked_sum.assert_called_once_with(25, "cpu")
    # Single-rank returns the local count directly.
    assert train_mod._global_train_records(str(path), 1, False, SimpleNamespace(count=101, rank=None)) == 101
