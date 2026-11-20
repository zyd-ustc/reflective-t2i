"""Unit tests for the unified 2-arm bench_t2i_closed_loop64.py."""

import json

from scripts.bench_t2i_closed_loop64 import (
    _load_ref_module,
    build_t2i_cases,
    classify_diagnosed_action,
    cmd_merge,
    parse_geneval_pp_response,
    parse_yesno_token,
)


def test_classify_diagnosed_action_and_parsers():
    assert classify_diagnosed_action("Add 2 zebras to the image so that it contains 5 zebras.") == "add"
    assert classify_diagnosed_action("Remove 1 umbrella from the right side.") == "remove"
    assert classify_diagnosed_action("Move the dog to the left of the chair.") == "move"
    assert classify_diagnosed_action("Change the color of the car to blue.") == "attr"
    assert classify_diagnosed_action("") == "none"

    assert parse_yesno_token("yes") == "yes"
    assert parse_yesno_token("<think>counting...</think>\nNo.") == "no"
    assert parse_yesno_token("unclear") is None

    gv_mod = _load_ref_module(
        "_gv_test",
        "qwen_latent_cot/benchmarks/reference/scoring_sources/bench_tools/benchmarks/geneval/scripts/eval_geneval_qwen35.py",
    )
    pp_mod = _load_ref_module(
        "_pp_test",
        "qwen_latent_cot/benchmarks/reference/scoring_sources/pp/eval_geneval_pp_qwen.py",
    )
    qs = gv_mod.build_questions(
        {
            "geneval_include": [{"class": "apple", "count": 3, "color": "red"}],
            "geneval_exclude": [{"class": "apple", "count": 4}],
        }
    )
    assert len(qs) == 2
    assert qs[0]["must_be"] == "yes"
    assert qs[1]["must_be"] == "no"

    res = parse_geneval_pp_response('```json\n{"correct": 1, "reason": "all matched"}\n```', pp_mod)
    assert res == {"correct": 1, "reason": "all matched"}
    res2 = parse_geneval_pp_response('<think>check</think>{"correct": false, "reason": "wrong count"}', pp_mod)
    assert res2["correct"] == 0


def test_build_cases_and_merge_funnel(tmp_path):
    gv_file = tmp_path / "gv.jsonl"
    pp_file = tmp_path / "pp.jsonl"
    gv_file.write_text(
        json.dumps(
            {
                "record_id": "gv_0",
                "geneval_prompt": "three dogs",
                "geneval_tag": "counting",
                "seed": 42,
                "geneval_include": [{"class": "dog", "count": 3}],
                "geneval_exclude": [{"class": "dog", "count": 4}],
            }
        )
        + "\n"
    )
    pp_file.write_text(
        json.dumps(
            {
                "record_id": "pp_0",
                "instruction": "two red cups on the left",
                "required_prompt": "two red cups on the left",
                "category": "color_spatial_attr",
                "seed": 99,
                "benchmark_metadata": {"include": [{"class": "cup", "count": 2, "color": "red"}]},
            }
        )
        + "\n"
    )

    built = build_t2i_cases(gv_file, pp_file)
    assert built["total_cases"] == 2
    assert built["benchmarks"] == {"geneval": 1, "geneval_pp": 1}

    out = tmp_path / "bench_out"
    (out / "gen_shards").mkdir(parents=True)
    (out / "score_shards").mkdir(parents=True)
    (out / "t2i_cases.json").write_text(json.dumps(built))

    gen_rows = [
        {
            "case_id": "geneval:gv_0",
            "benchmark": "geneval",
            "idx_in_bench": 0,
            "record_id": "gv_0",
            "category": "counting",
            "prompt": "three dogs",
            "seed": 42,
            "checkpoint_step": 32000,
            "read_time": 0.36,
            "stop_time": 0.60,
            "lookahead_steps": 4,
            "edit_cfg_scale": 1.0,
            "outcome": "edit",
            "diagnosed_action": "add",
            "instruction": "Add 1 dog.",
            "gated_by": None,
            "rechecked_add_subject": "dog",
            "strip_path": "strips/gv_0.jpg",
            "arms": {
                "baseline": {"triggered": False, "image_rel": "a.png", "total_nfe": 99, "extra_nfe": 0, "total_seconds": 10.0, "oob_frac": 0.02, "hf_grad": 0.05},
                "isochronic": {"triggered": True, "image_rel": "b.png", "total_nfe": 112, "extra_nfe": 13, "total_seconds": 15.0, "oob_frac": 0.02, "hf_grad": 0.05},
            },
        },
        {
            "case_id": "geneval_pp:pp_0",
            "benchmark": "geneval_pp",
            "idx_in_bench": 0,
            "record_id": "pp_0",
            "category": "color_spatial_attr",
            "prompt": "two red cups on the left",
            "seed": 99,
            "checkpoint_step": 32000,
            "read_time": 0.36,
            "stop_time": 0.60,
            "lookahead_steps": 4,
            "edit_cfg_scale": 1.0,
            "outcome": "satisfied",
            "diagnosed_action": "none",
            "instruction": "",
            "gated_by": "focused_add_recheck",
            "rechecked_add_subject": "cup",
            "strip_path": None,
            "arms": {
                "baseline": {"triggered": False, "image_rel": "p0.png", "total_nfe": 99, "extra_nfe": 0, "total_seconds": 10.0, "oob_frac": 0.02, "hf_grad": 0.05},
                "isochronic": {"triggered": False, "image_rel": "p0.png", "total_nfe": 107, "extra_nfe": 8, "total_seconds": 12.0, "oob_frac": 0.02, "hf_grad": 0.05},
            },
        },
    ]
    score_rows = [
        {
            "case_id": "geneval:gv_0",
            "benchmark": "geneval",
            "record_id": "gv_0",
            "category": "counting",
            "scores": {
                "baseline": {"status": "scored", "correct": 0},
                "isochronic": {"status": "scored", "correct": 1},
            },
        },
        {
            "case_id": "geneval_pp:pp_0",
            "benchmark": "geneval_pp",
            "record_id": "pp_0",
            "category": "color_spatial_attr",
            "scores": {
                "baseline": {"status": "scored", "correct": 1},
                "isochronic": {"status": "scored", "correct": 1},
            },
        },
    ]
    (out / "gen_shards" / "gen_rank00.jsonl").write_text("\n".join(json.dumps(r) for r in gen_rows) + "\n")
    (out / "score_shards" / "score_rank00.jsonl").write_text("\n".join(json.dumps(r) for r in score_rows) + "\n")

    class _Args:
        pass

    args = _Args()
    args.out = str(out)
    args.allow_partial = False
    cmd_merge(args)

    summary = json.loads((out / "summary.json").read_text())
    assert summary["integrity"]["complete"] is True
    assert summary["geneval"]["arms"]["baseline"]["micro_accuracy"] == 0.0
    assert summary["geneval"]["arms"]["isochronic"]["micro_accuracy"] == 1.0
    assert summary["geneval"]["arms"]["isochronic"]["fixed_on_trig_fail"] == 1
    assert summary["geneval_pp"]["gated_counts"] == {"focused_add_recheck": 1}
    assert (out / "summary.md").exists()
    assert (out / "index.html").exists()
