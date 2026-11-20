"""Offline tests for the count/spatial pair builder.

Every test runs without a server: the vision model and the editor are stubs,
and the manifest is checked by the Stage 1 loader itself.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from qwen_latent_cot.count_spatial.candidates import _split, read_index, select
from qwen_latent_cot.count_spatial.editor import EditClient, EditError
from qwen_latent_cot.count_spatial.manifest import record, validate, write_shards
from qwen_latent_cot.count_spatial.observe import check_observation, countable, suitable
from qwen_latent_cot.count_spatial.plan import COUNT_INCREASE, COUNT_REDUCE, SPATIAL, choose, family_counts
from qwen_latent_cot.count_spatial.verify import check as verify_pair
from qwen_latent_cot.count_spatial.verify import instruction
from qwen_latent_cot.count_spatial.vlm import VlmClient, VlmError, open_traces


CONFIG = Path(__file__).resolve().parent.parent / "configs" / "u15_isochronic_state" / "isochronic_manifold.json"


def observation(objects, relations=(), quality="good", candidates=None):
    """An observation in the shape the prompt asks for."""
    objects = [dict(name=name, count=count, positions=list(positions), countable=True)
               for name, count, positions in objects]
    if candidates is None:
        candidates = [dict(object=obj["name"], count=obj["count"]) for obj in objects if obj["count"] >= 2]
    return dict(objects=objects, layout="the scene as observed",
                count_candidates=candidates,
                spatial_relations=[dict(subject=s, relation=r, reference=ref, clear=True)
                                   for s, r, ref in relations],
                quality=quality, quality_notes="")


APPLES = observation([("apple", 3, ["left", "center", "right"]), ("plate", 1, ["center"])],
                     relations=[("apple", "left of", "plate")], candidates=[dict(object="apple", count=3)])


def test_observation_shape_is_enforced():
    check_observation(APPLES)
    with pytest.raises(ValueError):
        check_observation(dict(APPLES, quality="terrible"))
    with pytest.raises(ValueError):
        check_observation({k: v for k, v in APPLES.items() if k != "count_candidates"})


def test_count_candidate_needs_two_visible_instances_and_an_agreeing_count():
    assert [c["object"] for c in countable(APPLES)] == ["apple"]
    disputed = observation([("apple", 3, ["left"])], candidates=[dict(object="apple", count=5)])
    assert countable(disputed) == []
    single = observation([("apple", 1, ["left"])])
    assert not suitable(single)


def test_the_plan_names_the_instance_not_just_a_count():
    plan = choose("/x.png", APPLES, 0)
    assert plan["edit_family"] in (COUNT_REDUCE, COUNT_INCREASE, SPATIAL)
    reduce_plans = []
    for index in range(3):
        found = choose("/x.png", APPLES, index)
        if found["edit_family"] == COUNT_REDUCE:
            reduce_plans.append(found)
    assert reduce_plans, "the reduction family is reachable from this observation"
    assert "the one at the right" in reduce_plans[0]["breakage_instruction"]
    assert reduce_plans[0]["count_source"] == 2 and reduce_plans[0]["count_predicted"] == 3


def test_increase_plan_creates_a_larger_target_from_the_original():
    plan = next(p for p in (choose("/x.png", APPLES, i) for i in range(3))
                if p["edit_family"] == COUNT_INCREASE)
    assert plan["count_source"] == 3 and plan["count_predicted"] == 4
    assert plan["breakage_instruction"].startswith("Add one")
    assert plan["count_direction"] == "increase"


def test_quotas_follow_the_plan_mix_and_sum_to_the_limit():
    quotas = family_counts(5000)
    assert sum(quotas.values()) == 5000
    assert quotas[COUNT_REDUCE] == quotas[COUNT_INCREASE] == 2000
    assert quotas[SPATIAL] == 1000


def test_quota_zero_stops_the_family_from_being_planned():
    observation_spatial = observation([("cup", 1, ["left"]), ("table", 1, ["right"])],
                                      relations=[("cup", "left of", "table")])
    assert choose("/x.png", observation_spatial, 0, {SPATIAL: 1, COUNT_REDUCE: 0, COUNT_INCREASE: 0}) is not None
    assert choose("/x.png", observation_spatial, 0, {SPATIAL: 0, COUNT_REDUCE: 0, COUNT_INCREASE: 0}) is None


def reduce_plan(scene=APPLES):
    """The reduction plan for a scene, whichever rotation order offers it first."""
    for index in range(3):
        plan = choose("/x.png", scene, index)
        if plan is not None and plan["edit_family"] == COUNT_REDUCE:
            return plan
    raise AssertionError("the reduction family was never planned")


def test_verification_fails_closed_when_a_count_is_missing():
    plan = reduce_plan()
    wrong = observation([("apple", 2, ["left"]), ("plate", 1, ["center"])])
    verdict = verify_pair(dict(target=APPLES, wrong=wrong), plan)
    assert verdict["passed"]
    unseen = observation([("plate", 1, ["center"])])
    assert not verify_pair(dict(target=APPLES, wrong=unseen), plan)["passed"]


def test_verification_rejects_an_unrelated_object_changing():
    plan = reduce_plan()
    wrong = observation([("apple", 2, ["left"]), ("plate", 2, ["center"])])
    verdict = verify_pair(dict(target=APPLES, wrong=wrong), plan)
    assert not verdict["passed"]
    assert verdict["unrelated_count_changes"] == ["plate"]


def test_verification_rejects_missing_unrelated_objects():
    plan = reduce_plan()
    wrong = observation([("apple", 2, ["left"])])
    verdict = verify_pair(dict(target=APPLES, wrong=wrong), plan)
    assert not verdict["passed"]
    assert "plate" in verdict["dropped_objects"]


def test_instruction_uses_the_measured_count_not_the_predicted_one():
    plan = reduce_plan()
    wrong = observation([("apple", 2, ["left"]), ("plate", 1, ["center"])])
    verdict = verify_pair(dict(target=APPLES, wrong=wrong), plan)
    assert instruction(verdict) == (
        "Add 1 apple to the image so that it contains exactly 3 apple instead of the current 2, "
        "placing the new ones in plausible empty positions; keep everything else unchanged.")


def test_a_spatial_plan_moves_between_known_sides():
    scene = observation([("cup", 1, ["left"]), ("plate", 1, ["right"])],
                        relations=[("cup", "left of", "plate")])
    plan = choose("/x.png", scene, 0)
    assert plan["edit_family"] == SPATIAL
    assert plan["subject_side"] == "left" and plan["destination"] == "right"
    assert "Move only the cup to the right side of the plate" in plan["breakage_instruction"]
    assert "plate must stay exactly in its original position" in plan["breakage_instruction"]


def test_vertical_relation_is_not_rewritten_as_a_horizontal_move():
    scene = observation([("cup", 1, ["left"]), ("plate", 1, ["right"])],
                        relations=[("cup", "above", "plate")])
    plan = choose("/x.png", scene, 0)
    assert "to the above" not in plan["breakage_instruction"]
    assert "below the plate" in plan["breakage_instruction"]
    wrong = observation([("cup", 1, ["left"]), ("plate", 1, ["right"])],
                        relations=[("plate", "above", "cup")])
    verdict = verify_pair({"target": scene, "wrong": wrong}, plan)
    assert verdict["passed"]
    assert instruction(verdict) == "Move the cup above the plate; keep the rest of the image unchanged."


def test_spatial_verification_uses_relative_relation_and_writes_instruction():
    target = observation([("cup", 1, ["left"]), ("plate", 1, ["right"])],
                         relations=[("cup", "left of", "plate")])
    plan = choose("/x.png", target, 0)
    wrong = observation([("cup", 1, ["right"]), ("plate", 1, ["center"])],
                        relations=[("plate", "left of", "cup")])
    verdict = verify_pair(dict(target=target, wrong=wrong), plan)
    assert verdict["passed"]
    assert instruction(verdict) == "Move the cup to the left of the plate; keep the rest of the image unchanged."
    # A movement in absolute image coordinates is not a relation reversal.
    unchanged = observation([("cup", 1, ["right"]), ("plate", 1, ["right"])],
                            relations=[("plate", "right of", "cup")])
    assert not verify_pair(dict(target=target, wrong=unchanged), plan)["passed"]


def test_spatial_verification_rejects_a_reference_that_moves():
    target = observation([("cup", 1, ["left"]), ("plate", 1, ["right"])],
                         relations=[("cup", "left of", "plate")])
    wrong = observation([("cup", 1, ["right"]), ("plate", 1, ["left"])],
                        relations=[("cup", "right of", "plate")])
    verdict = verify_pair({"target": target, "wrong": wrong}, choose("/x.png", target, 0))
    assert not verdict["passed"]
    assert "reference object stayed in place" in verdict["reasons"]


def test_pair_observations_do_not_require_observation_only_candidates():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import count_spatial_edits as cli
    pair_observation = {k: v for k, v in APPLES.items() if k != "count_candidates"}
    cli.check_pair({"target": pair_observation, "wrong": pair_observation})


class _StubTransport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, max_tokens):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_vlm_retries_a_malformed_reply_then_succeeds(tmp_path):
    client_traces = open_traces(tmp_path / "trace", "observe")
    client = VlmClient(transport=_StubTransport(["not json at all", json.dumps(APPLES)]), retries=2,
                       traces=client_traces)
    assert client.call([{"role": "user", "content": "x"}], check=check_observation,
                       stage="observe") == APPLES
    for handle in client_traces.values():
        handle.close()
    lines = [json.loads(line) for line in (tmp_path / "trace" / "observe.jsonl").read_text().splitlines()]
    assert [line["ok"] for line in lines] == [False, True]


def test_vlm_gives_up_after_the_retry_budget():
    client = VlmClient(transport=_StubTransport(["{}", "{}", "{}"]), retries=3)
    with pytest.raises(VlmError):
        client.call([{"role": "user", "content": "x"}], check=check_observation)


def _png(path, size=(512, 512), colour=(10, 120, 80)):
    Image.new("RGB", size, colour).save(path)
    return path


def test_editor_rejects_a_reply_without_an_image(tmp_path):
    source = _png(tmp_path / "src.png")
    client = EditClient("http://x", "m", transport=lambda fields, files: {"data": [{}]})
    with pytest.raises(EditError):
        client.edit(source, "Remove one apple.")


def test_editor_rejects_an_aspect_ratio_the_pair_cannot_carry(tmp_path):
    import base64
    import io

    source = _png(tmp_path / "src.png", (512, 512))
    other = io.BytesIO()
    Image.new("RGB", (768, 512)).save(other, format="PNG")
    reply = {"data": [{"b64_json": base64.b64encode(other.getvalue()).decode()}]}
    client = EditClient("http://x", "m", transport=lambda fields, files: reply)
    with pytest.raises(EditError):
        client.edit(source, "Remove one apple.")


def test_candidates_skip_unsupported_benchmarks_and_extreme_ratios(tmp_path):
    good = _png(tmp_path / "a.png")
    _png(tmp_path / "wide.png", (1000, 300), (5, 5, 5))
    index = tmp_path / "index.csv"
    index.write_text(
        "uid,benchmark,wh_ratio,width,height,image\n"
        "ok,geneval2,1:1,512,512,%s\n"
        "other,unigenbench,1:1,512,512,%s\n"
        "wide,dpg,10:3,1000,300,%s\n"
        "gone,dpg,1:1,512,512,%s\n" % (good, good, tmp_path / "wide.png", tmp_path / "missing.png"))
    rows = list(read_index(index))
    assert [row["uid"] for row in rows] == ["ok"]
    out = tmp_path / "candidates.jsonl"
    assert select(index, 10, out, seed=1)["written"] == 1


def test_select_resume_keeps_the_trial_assignment(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import count_spatial_edits as cli
    image = _png(tmp_path / "source.png")
    index = tmp_path / "index.csv"
    index.write_text("uid,benchmark,wh_ratio,width,height,image\n" +
                     "".join("%s,dpg,1:1,512,512,%s\n" % (n, image) for n in range(4)))
    cfg = {"out": str(tmp_path / "trial"), "index_csv": str(index), "limit": 2}
    cli.cmd_select(cfg)
    original = (tmp_path / "trial" / "candidates.jsonl").read_text()
    cli.cmd_select(cfg)
    assert (tmp_path / "trial" / "candidates.jsonl").read_text() == original
    assert len(cli.read_jsonl(tmp_path / "trial" / "trials.jsonl")) == 1


def test_split_is_stable_and_family_shaped():
    assert _split("abc") == _split("abc")
    buckets = {_split(str(i)) for i in range(200)}
    assert buckets <= {"train", "val", "test"}


def _candidate(uid):
    return dict(uid=uid, benchmark="dpg", source_family=uid, prompt_family=uid)


def _needs_stage1_loader():
    try:
        import torch  # noqa: F401
    except Exception as exc:  # a broken torch wheel, not a missing one
        return False, "the Stage 1 loader needs torch: %s" % str(exc).splitlines()[0]
    return True, ""


@pytest.mark.skipif(not _needs_stage1_loader()[0], reason=_needs_stage1_loader()[1])
def test_manifest_is_accepted_by_the_stage1_loader(tmp_path):
    plan = reduce_plan()
    wrong = observation([("apple", 2, ["left"]), ("plate", 1, ["center"])])
    verdict = verify_pair(dict(target=APPLES, wrong=wrong), plan)
    text = instruction(verdict)

    records = []
    for number in range(3):
        uid = "uid%d" % number
        candidate = _candidate(uid)
        target = _png(tmp_path / (uid + "_target.png"))
        source = _png(tmp_path / (uid + "_wrong.png"))
        records.append(record(candidate, plan, verdict, text, source, target, "three apples on a plate"))
    shards = write_shards(records, tmp_path / "out", shard_size=2)
    assert len(shards) == 2
    assert sum(len(path.read_text().splitlines()) for path in shards) == 3

    report = validate(shards, CONFIG)
    assert report["pairs"] == 3
    row = json.loads(shards[0].read_text().splitlines()[0])
    assert row["condition_protocol"] == "semantic_feedback_only_v1"
    assert row["semantic_task"] == COUNT_REDUCE
    assert row["count_direction"] == "increase"
    assert row["role"] == "L_edit" and row["semantic_approved"] is True
    assert row["split"] in ("train", "val", "test")


@pytest.mark.skipif(not _needs_stage1_loader()[0], reason=_needs_stage1_loader()[1])
def test_manifest_loader_rejects_a_missing_instruction(tmp_path):
    plan = reduce_plan()
    verdict = verify_pair(dict(target=APPLES, wrong=observation([("apple", 2, ["left"])])), plan)
    candidate = _candidate("bad")
    target = _png(tmp_path / "t.png")
    source = _png(tmp_path / "s.png")
    row = record(candidate, plan, verdict, "", source, target, "caption")
    shards = write_shards([row], tmp_path / "bad")
    with pytest.raises(ValueError):
        validate(shards, CONFIG)


# ---- the command-line steps, end to end, with stub services ---------------

def _trial_config(tmp_path, index_csv):
    image = _png(tmp_path / "correct.png")
    rows = [dict(uid="uid_%03d" % n, benchmark="dpg", image=str(image), width=512, height=512,
                 source_family="uid_%03d" % n, prompt_family="uid_%03d" % n)
            for n in range(6)]
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    from qwen_latent_cot.count_spatial.manifest import split_for
    config = tmp_path / "trial.json"
    config.write_text(json.dumps(dict(
        out=str(tmp_path / "out"), candidates=str(candidates), index_csv=str(index_csv),
        limit=6, workers=2, retries=1, config=str(CONFIG), shard_size=2,
        family_weights={"count_reduce": 0.4, "count_increase": 0.4, "spatial": 0.2},
        vlm_url="http://stub/v1/chat/completions", editor=dict(base_url="http://stub"),
    )))
    return config


@pytest.mark.skipif(not _needs_stage1_loader()[0], reason=_needs_stage1_loader()[1])
@pytest.mark.parametrize("verification_mode", ["pair", "independent"])
def test_steps_carry_a_record_from_candidate_to_loader_accepted_manifest(tmp_path, monkeypatch, verification_mode):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import count_spatial_edits as cli
    import qwen_latent_cot.count_spatial.editor as editor_mod

    config = _trial_config(tmp_path, tmp_path / "unused.csv")
    cfg = cli.load_config(config)
    cfg["verification_mode"] = verification_mode

    class _Vlm:
        """Plays the observe call now and, later, the pair call per plan family."""
        verification_uids = []

        def __init__(self, url="", model="", **kwargs):
            self.stage = None

        def call_many(self, jobs):
            out, planned = {}, {}
            if (Path(cfg["out"]) / "plans.jsonl").exists():
                for line in (Path(cfg["out"]) / "plans.jsonl").read_text().splitlines():
                    row = json.loads(line)
                    planned[row["uid"]] = row["plan"]
            for uid, messages, kwargs in jobs:
                if kwargs.get("stage") == "observe":
                    out[uid] = (APPLES, None)
                    continue
                self.verification_uids.append(uid)
                if "::" in uid:
                    original_uid, side = uid.rsplit("::", 1)
                    out[uid] = (self.after(planned[original_uid])[side], None)
                else:
                    out[uid] = (self.after(planned[uid]), None)
            return out

        @staticmethod
        def after(plan):
            """The picture the editor was asked for, as the verifier would see it."""
            if plan["edit_family"] == "count_increase":
                larger = observation([("apple", 4, ["left", "center", "right"]), ("plate", 1, ["center"])])
                return dict(target=larger, wrong=APPLES)
            if plan["edit_family"] == "count_reduce":
                # Whichever family planned it, the pair the pipeline can keep is
                # one the instruction "Add N" describes: the wrong image holds
                # fewer apples than the verified target.
                return dict(target=APPLES, wrong=observation([("apple", 2, ["left"]), ("plate", 1, ["center"])]))
            # The inverse phrasing records apple right of plate.
            return dict(target=APPLES,
                        wrong=observation([("apple", 3, ["left", "center", "right"]), ("plate", 1, ["left"])],
                                          relations=[("plate", "left of", "apple")]))

    monkeypatch.setattr(cli, "make_vlm", lambda cfg_, root, stage, transport=None, client=None: _Vlm())
    cli.cmd_observe(cfg)
    cli.cmd_plan(cfg)

    def fake_edit(self, source, instruction, seed=42):
        from PIL import Image
        return dict(image=Image.new("RGB", (512, 512), (0, 0, 0)), seconds=0.1, seed=seed, size=[512, 512])

    monkeypatch.setattr(editor_mod.EditClient, "edit", fake_edit)
    cli.cmd_edit(cfg)
    for row in cli.read_jsonl(Path(cfg["out"]) / "plans.jsonl"):
        if row["plan"]["edit_family"] == "count_increase":
            images = Path(cfg["out"]) / "images"
            assert (images / (row["uid"] + "_wrong.png")).is_symlink()
            assert not (images / (row["uid"] + "_target.png")).is_symlink()
    cli.cmd_verify(cfg)
    assert len(_Vlm.verification_uids) == (12 if verification_mode == "independent" else 6)
    report = cli.cmd_manifest(cfg)
    assert report["pairs"] == 6
    assert report["train"] + report["splits"].get("val", 0) + report["splits"].get("test", 0) == 6
    out = Path(cfg["out"])
    assert (out / "STATUS.md").exists() is False  # report step writes it, not manifest
    status = cli.cmd_report(cfg)
    assert status["counts"]["verified"] == 6
    assert (out / "STATUS.md").exists()


def test_hosted_verification_is_refused_unless_allowed(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import count_spatial_edits as cli

    config = _trial_config(tmp_path, tmp_path / "unused.csv")
    cfg = cli.load_config(config)
    cfg["vlm_kind"] = "glm"
    with pytest.raises(SystemExit):
        cli.cmd_verify(cfg)
    cfg["allow_hosted_verify"] = True
    cfg["vlm_kind"] = "local"
    # With the flag set and no edited images yet, the step is a no-op rather than a refusal.
    assert cli.cmd_verify(cfg)["pending"] == 0


def test_the_hosted_client_requires_a_key(monkeypatch):
    from qwen_latent_cot.count_spatial.glm import GlmClient, resolve_glm_endpoint, resolve_glm_key

    monkeypatch.delenv("METTO_GLM_KEY", raising=False)
    monkeypatch.delenv("METTO_KEY", raising=False)
    monkeypatch.setenv("METTO_BASE_URL", "https://metto-api.com")
    with pytest.raises(ValueError):
        GlmClient("")
    assert resolve_glm_endpoint("http://127.0.0.1:8092/v1/chat/completions") == "https://metto-api.com/v1/chat/completions"
    assert resolve_glm_endpoint("https://metto-api.com") == "https://metto-api.com/v1/chat/completions"

    monkeypatch.setenv("METTO_GLM_KEY", "sk-test-glm")
    assert resolve_glm_key("METTO_GLM_KEY") == "sk-test-glm"
    assert VlmClient._decode('<think>reasoning</think>\n```json\n{"ok": 1}\n```') == {"ok": 1}

    # Count candidates > 8 (e.g. 100 tiny stars) are skipped
    crowd = observation([("star", 100, ["center"])], candidates=[dict(object="star", count=100)])
    assert countable(crowd) == []


def test_instruction_branches_on_the_edit_family_not_the_task():
    """Both count families share task "count"; both must get the count sentence."""
    from qwen_latent_cot.count_spatial.verify import instruction as write

    # Measured counts as verification reports them: the wrong image always
    # holds fewer objects than the target, so "Add N ... exactly M".
    cases = (("count_reduce", 2, 3), ("count_increase", 3, 4))
    for family, source, goal in cases:
        verdict = dict(passed=True, target_count=goal, wrong_count=source,
                       plan=dict(edit_family=family, semantic_task=family, task="count", object="apple"))
        text = write(verdict)
        assert text == ("Add 1 apple to the image so that it contains exactly %d apple instead of the "
                        "current %d, placing the new ones in plausible empty positions; "
                        "keep everything else unchanged." % (goal, source)), (family, text)

    spatial = dict(passed=True, target={"objects": [{"name": "cup", "positions": ["left"], "count": 1}],
                                        "spatial_relations": []},
                   plan=dict(edit_family="spatial", semantic_task="spatial", task="spatial",
                             subject="cup", relation="left of", relation_target="right of", reference="plate"))
    assert write(spatial) == "Move the cup to the left of the plate; keep the rest of the image unchanged."


def test_task_coverage_verification_skips_pixel_inventory_checks():
    # Count edit where removing a man also removes the goblet in his hand
    scene = observation([("man", 2, ["left", "right"]), ("goblet", 1, ["right"]), ("woman", 1, ["bottom right"])])
    plan = reduce_plan(scene)
    wrong = observation([("man", 1, ["left"]), ("woman", 1, ["bottom right"])])
    assert not verify_pair({"target": scene, "wrong": wrong}, plan, strict_preservation=True)["passed"]
    verdict = verify_pair({"target": scene, "wrong": wrong}, plan, strict_preservation=False)
    assert verdict["passed"]
    assert instruction(verdict).startswith("Add 1 man to the image so that it contains exactly 2 man instead of the current 1")

    # Spatial edit where badger flips across backpack while backpack shifts slightly across grid cells
    t_sp = observation([("badger", 1, ["right"]), ("backpack", 1, ["left"])], relations=[("badger", "right of", "backpack")])
    p_sp = choose("/x.png", t_sp, 0)
    w_sp = observation([("badger", 1, ["left"]), ("backpack", 1, ["right"])], relations=[("badger", "left of", "backpack")])
    assert not verify_pair({"target": t_sp, "wrong": w_sp}, p_sp, strict_preservation=True)["passed"]
    v_sp = verify_pair({"target": t_sp, "wrong": w_sp}, p_sp, strict_preservation=False)
    assert v_sp["passed"]
    assert instruction(v_sp) == "Move the badger to the right of the backpack; keep the rest of the image unchanged."

    # Still fails closed if reference object was deleted
    deleted_ref = observation([("badger", 1, ["left"])], relations=[])
    assert not verify_pair({"target": t_sp, "wrong": deleted_ref}, p_sp, strict_preservation=False)["passed"]
