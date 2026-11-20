"""Build count and spatial edit pairs backwards from verified T2I images.

The correct image is the anchor and the training target. An editor makes it
wrong, a vision model observes both pictures, and the training instruction is
written from the difference that was actually measured.

    python scripts/count_spatial_edits.py configs/count_spatial_20261115/trial.json
    python scripts/count_spatial_edits.py configs/count_spatial_20261115/trial.json select observe

Each step is a command of its own, so a trial can be inspected and resumed
between steps:

    select    Step 1  pick candidates from the T2I index
    observe   Step 2  what each correct image shows
    plan      Step 3  the breakage and the predicted repair
    edit      Step 4  generate the wrong source images
    verify    Step 5+6  check the pair and write the instruction
    manifest  Step 7  shards the Stage 1 loader accepts
    report    read the STATUS.md numbers back

A step re-runs only the records it has not finished, and every record it drops
is written to ``rejects.jsonl`` with the reason, so throughput is measurable
rather than assumed.
"""

import argparse
import base64
import json
import io
import os
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from qwen_latent_cot.count_spatial import candidates as cand_mod
from qwen_latent_cot.count_spatial import glm as glm_adapter
from qwen_latent_cot.count_spatial import vlm as vlm_mod
from qwen_latent_cot.count_spatial.editor import EditClient, EditError
from qwen_latent_cot.count_spatial.vlm import VlmClient
from qwen_latent_cot.count_spatial.observe import OBSERVE_PROMPT, check_observation, suitable
from qwen_latent_cot.count_spatial.plan import FAMILIES, choose, family_counts
from qwen_latent_cot.count_spatial.verify import check as verify_pair
from qwen_latent_cot.count_spatial.verify import instruction

OBSERVE_PAIR_PROMPT = """You are inspecting two pictures of the same scene for a data pipeline. Describe each picture independently, including any differences you can see.

Report only what you can see in each picture; never infer an object or a relation from general knowledge of such scenes.

Return JSON only, with one observation per picture:
{{
  "target": <observation of the FIRST picture>,
  "wrong": <observation of the SECOND picture>
}}

Each observation has exactly this shape:
{{
  "objects": [{{"name": "<singular noun, lower case>", "count": <int>, "positions": ["<POSITION>..."],
               "countable": <true if you can count each instance separately>}}],
  "layout": "<one sentence describing where things are>",
  "spatial_relations": [{{"subject": "<name>", "relation": "left of|right of|above|below|on|inside",
                         "reference": "<name>", "clear": <true if the relation is unambiguous>}}],
  "quality": "<good|acceptable|poor>",
  "quality_notes": "<what is wrong with the picture, if anything>"
}}

POSITION must be one of: {positions}.
Report every object you can clearly see, including background ones, and give each one the same name in both pictures when it is the same thing. Group instances of the same object into one entry with their total count. Keep layout and quality_notes short. Return compact JSON without indentation. If a picture is blurry, empty, mostly one colour, or otherwise unusable, say so with quality set to poor.
"""


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    base = Path(path).parent / "base.json"
    if cfg.pop("extends", None) and base.exists():
        merged = json.loads(base.read_text())
        merged.update(cfg)
        cfg = merged
    # An inherited trainer config names a file in the repo, not beside this one.
    for key, relative_to in (("out", "here"), ("candidates", "here"), ("config", "repo")):
        if key in cfg and not Path(cfg[key]).is_absolute():
            base = Path(path).parent if relative_to == "here" else REPO
            cfg[key] = str((base / cfg[key]).resolve())
    return cfg


def paths(cfg):
    root = Path(cfg["out"])
    return dict(
        root=root,
        trials=Path(cfg["trials"]) if cfg.get("trials") else root / "trials.jsonl",
        candidates=Path(cfg["candidates"]) if cfg.get("candidates") else root / "candidates.jsonl",
        observations=Path(cfg["observations"]) if cfg.get("observations") else root / "observations.jsonl",
        plans=root / "plans.jsonl",
        images=root / "images",
        verified=root / "verified.jsonl",
        rejects=root / "rejects.jsonl",
        manifest=root,
        status=root / "STATUS.md",
        traces=root / "logs" / "vlm",
    )


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_appender(path):
    """Append-and-flush per record, so an interrupted step keeps what it finished."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    tmp.replace(path)


class Rejects:
    """Append-only log of why a uid left the pipeline; the trial's real cost."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, uid, step, reason):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(uid=uid, step=step, reason=reason)) + "\n")
        return None


def make_vlm(cfg, root, stage, transport=None, client=VlmClient):
    """The vision client for one stage: local by default, hosted when configured."""
    return glm_adapter.make(cfg, root, stage, transport=transport, client=client)


def image_message(path, max_side=1024):
    """One JPEG-encoded picture; the observation never needs the PNG bytes."""
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()}}


def observe_messages(path):
    return [{"role": "user", "content": [{"type": "text", "text": OBSERVE_PROMPT},
                                         image_message(path)]}]


def verify_messages(target_path, wrong_path, plan=None):
    """Both pictures in one call: the difference is compared, not described twice."""
    from qwen_latent_cot.count_spatial.observe import POSITIONS
    prompt = OBSERVE_PAIR_PROMPT.format(positions=", ".join(POSITIONS))
    if plan:
        nouns = [plan.get("object")] if plan.get("task") == "count" else [plan.get("subject"), plan.get("reference")]
        nouns = [n for n in nouns if n]
        if nouns:
            prompt += (
                "\nVocabulary alignment: when those objects are visible, refer to them using the exact singular "
                "noun(s) %s rather than synonyms, while counting and locating strictly what you see in each picture."
                % ", ".join(repr(n) for n in nouns)
            )
    return [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        image_message(target_path), image_message(wrong_path)]}]


def check_pair(value):
    """A pair reply must carry one well-formed observation per picture."""
    for key in ("target", "wrong"):
        if key not in value:
            raise ValueError("the pair reply lacks " + key)
        check_observation(value[key], require_candidates=False)


def require_local_verifier(cfg):
    """Verification must not run on the hosted model while local is configured to miss."""
    kind = cfg.get("vlm_kind", "local")
    if kind == "glm" and not cfg.get("allow_hosted_verify", False):
        raise SystemExit(
            "vlm_kind is 'glm': verification would accept samples the local model is supposed "
            "to judge. Set allow_hosted_verify=true in the trial config to proceed anyway.")


# ---------------------------------------------------------------- commands

def cmd_select(cfg):
    """Draw the next slice of the library after every earlier trial's slice."""
    p = paths(cfg)
    trials = read_jsonl(p["trials"])
    existing = next((row for row in trials if row["out"] == str(p["root"])), None)
    if existing:
        if not p["candidates"].exists():
            raise ValueError("trial assignment exists but its candidates file is missing")
        return {**existing, "resumed": True}
    if p["candidates"].exists():
        raise ValueError("candidates file already exists without a trial assignment")
    skip = sum(row["written"] for row in trials)
    info = cand_mod.select(cfg["index_csv"], cfg["limit"], p["candidates"], seed=cfg.get("seed", 2026),
                           skip=skip)
    with jsonl_appender(p["trials"]) as handle:
        handle.write(json.dumps(dict(out=str(p["root"]), skip=skip, limit=cfg["limit"],
                                     written=info["written"], usable=info["usable"])) + "\n")
    return info


def cmd_observe(cfg, transport=None, vlm_client=VlmClient):
    p = paths(cfg)
    rows = read_jsonl(p["candidates"])
    done = {row["uid"] for row in read_jsonl(p["observations"])}
    todo = [row for row in rows if row["uid"] not in done]
    vlm = make_vlm(cfg, p["root"], "observe", transport, vlm_client)
    rejects = Rejects(p["rejects"])
    jobs = [(row["uid"], observe_messages(row["image"]), dict(check=check_observation, stage="observe", uid=row["uid"]))
            for row in todo]
    results = vlm.call_many(jobs)
    by_uid = {row["uid"]: row for row in rows}
    opened = 0
    with jsonl_appender(p["observations"]) as handle:
        for uid, (value, error) in results.items():
            if value is None:
                rejects.add(uid, "observe", error)
                continue
            if not suitable(value):
                rejects.add(uid, "observe", "not suitable: " + str(value.get("quality")) + " / "
                            + str(value.get("quality_notes")))
                continue
            handle.write(json.dumps(dict(uid=uid, benchmark=by_uid[uid]["benchmark"], image=by_uid[uid]["image"],
                                         observation=value)) + "\n")
            opened += 1
    return dict(candidates=len(rows), observed=len(todo), suitable=opened)


def cmd_plan(cfg):
    p = paths(cfg)
    observations = read_jsonl(p["observations"])
    weights = cfg.get("family_weights")
    quotas = family_counts(len(observations), weights) if weights else None
    plans = []
    for index, row in enumerate(observations):
        plan = choose(row["image"], row["observation"], index, quotas)
        if plan is None:
            Rejects(p["rejects"]).add(row["uid"], "plan", "nothing to break: no countable group or clear relation")
            continue
        if quotas is not None:
            quotas[plan["edit_family"]] -= 1
        plans.append(dict(uid=row["uid"], benchmark=row["benchmark"], image=row["image"], plan=plan,
                          observation=row["observation"]))
    write_jsonl(p["plans"], plans)
    mix = {family: sum(1 for item in plans if item["plan"]["edit_family"] == family) for family in FAMILIES}
    return dict(observed=len(observations), planned=len(plans), mix=mix)


def cmd_edit(cfg):
    """Ask the editor for every planned wrong image, concurrently."""
    p = paths(cfg)
    p["images"].mkdir(parents=True, exist_ok=True)
    plans = read_jsonl(p["plans"])
    rejects = Rejects(p["rejects"])
    editor = cfg["editor"]
    source_of = {row["uid"]: Path(row["image"]) for row in plans}
    instruction_of = {row["uid"]: row["plan"]["breakage_instruction"] for row in plans}
    family_of = {row["uid"]: row["plan"]["edit_family"] for row in plans}
    pending = [uid for uid in source_of if not all(
        (p["images"] / (uid + suffix)).exists() for suffix in ("_wrong.png", "_target.png"))]
    jobs = [(uid, source_of[uid], instruction_of[uid], editor.get("seed", 42)) for uid in pending]
    started, seconds = time.time(), []
    if pending:
        client = EditClient(editor["base_url"], editor.get("model", "Qwen/Qwen-Image-2.1"),
                            steps=editor.get("steps", 20), cfg_scale=editor.get("cfg_scale", 1.0),
                            timeout=editor.get("timeout", 300), workers=editor.get("workers", 1))
        results = client.edit_many(jobs)
        with p["images"].joinpath("edit.jsonl").open("a", encoding="utf-8") as log:
            for uid, (out, error) in results.items():
                if out is None:
                    rejects.add(uid, "edit", error)
                    continue
                # For count_increase the original is the smaller source and
                # the generated image is the larger target. Other families
                # start from the target and generate a wrong source.
                generated_suffix, original_suffix = (("_target.png", "_wrong.png")
                                                     if family_of[uid] == "count_increase"
                                                     else ("_wrong.png", "_target.png"))
                generated = p["images"] / (uid + generated_suffix)
                original = p["images"] / (uid + original_suffix)
                if not generated.exists():
                    out["image"].save(generated)
                if not original.exists():
                    original.symlink_to(source_of[uid].resolve())
                log.write(json.dumps(dict(uid=uid, seconds=out["seconds"], size=out["size"])) + "\n")
                seconds.append(out["seconds"])
            log.flush()
    return dict(planned=len(plans), pending=len(pending), edited=len(seconds),
                mean_seconds=round(sum(seconds) / len(seconds), 2) if seconds else None,
                elapsed=round(time.time() - started, 1))


def cmd_verify(cfg, transport=None, vlm_client=VlmClient):
    require_local_verifier(cfg)
    mode = cfg.get("verification_mode", "pair")
    if mode not in ("pair", "independent"):
        raise ValueError("verification_mode must be pair or independent")
    p = paths(cfg)
    plans = {row["uid"]: row for row in read_jsonl(p["plans"])}
    done = {row["uid"] for row in read_jsonl(p["verified"])}
    rejects = Rejects(p["rejects"])
    pending = []
    for uid, row in plans.items():
        wrong = p["images"] / (uid + "_wrong.png")
        if uid in done or not wrong.exists():
            continue
        target = p["images"] / (uid + "_target.png")
        if not target.exists():
            continue
        if mode == "independent":
            for side, path in (("target", target), ("wrong", wrong)):
                key = uid + "::" + side
                pending.append((key, observe_messages(path),
                                dict(check=check_observation, stage="verify", uid=key)))
        else:
            pending.append((uid, verify_messages(target, wrong, row["plan"]),
                            dict(check=check_pair, stage="verify", uid=uid)))
    vlm = make_vlm(cfg, p["root"], "verify", transport, vlm_client)
    results = vlm.call_many(pending)
    if mode == "independent":
        paired = {}
        for key, (value, error) in results.items():
            uid, side = key.rsplit("::", 1)
            pair = paired.setdefault(uid, {"values": {}, "errors": []})
            if error is not None or value is None:
                pair["errors"].append(side + ": " + str(error))
            else:
                pair["values"][side] = value
        results = {uid: (None, "; ".join(pair["errors"])) if pair["errors"]
                   else (pair["values"], None) for uid, pair in paired.items()}
    passed = 0
    with jsonl_appender(p["verified"]) as handle:
        for uid, (value, error) in results.items():
            if value is None:
                rejects.add(uid, "verify", error)
                continue
            row = plans[uid]
            verdict = verify_pair(value, row["plan"], strict_preservation=cfg.get("strict_preservation", True))
            text = instruction(verdict)
            if not verdict["passed"] or not text:
                verdict["reasons"] = verdict["reasons"] + ([] if text else ["no instruction could be written"])
                rejects.add(uid, "verify", "; ".join(verdict["reasons"]))
                continue
            handle.write(json.dumps(dict(uid=uid, benchmark=row["benchmark"], image=row["image"],
                                         plan=row["plan"], verdict=verdict, pair=value,
                                         instruction=text)) + "\n")
            passed += 1
    return dict(pending=len(results), verified=passed)


def cmd_manifest(cfg):
    p = paths(cfg)
    from qwen_latent_cot.count_spatial.manifest import record, validate, write_shards

    verified = read_jsonl(p["verified"])
    by_uid = {row["uid"]: row for row in read_jsonl(p["candidates"])}
    if not verified:
        return dict(shards=[], pairs=0, train=0, tail_rows=0, splits={})
    records = []
    for row in verified:
        # A verified record always has its candidate; a missing one is a bug here.
        candidate = by_uid[row["uid"]]
        condition = row.get("condition") or row.get("pair", {}).get("target", {}).get("layout", "")
        records.append(record(
            candidate, row["plan"], row["verdict"], row["instruction"],
            p["images"] / (row["uid"] + "_wrong.png"), p["images"] / (row["uid"] + "_target.png"),
            condition,
        ))
    shards = write_shards(records, p["manifest"], shard_size=cfg.get("shard_size", 500))
    report = validate(shards, cfg["config"])
    return dict(shards=[str(s) for s in shards], **report)


def cmd_report(cfg):
    p = paths(cfg)
    counts = dict(
        candidates=len(read_jsonl(p["candidates"])),
        observed=len(read_jsonl(p["observations"])),
        planned=len(read_jsonl(p["plans"])),
        edited=len(list(p["images"].glob("*_wrong.png"))) if p["images"].exists() else 0,
        verified=len(read_jsonl(p["verified"])),
    )
    reasons = {}
    for row in read_jsonl(p["rejects"]):
        reasons[row["step"]] = reasons.get(row["step"], 0) + 1
    mix = {}
    for row in read_jsonl(p["verified"]):
        mix[row["plan"]["edit_family"]] = mix.get(row["plan"]["edit_family"], 0) + 1
    lines = ["# count_spatial trial", "", "counts: " + json.dumps(counts), "", "rejects by step: "
             + json.dumps(reasons), "", "verified by family: " + json.dumps(mix)]
    p["status"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return dict(counts=counts, rejects=reasons, mix=mix)


COMMANDS = dict(select=cmd_select, observe=cmd_observe, plan=cmd_plan, edit=cmd_edit,
                verify=cmd_verify, manifest=cmd_manifest, report=cmd_report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the trial JSON")
    parser.add_argument("steps", nargs="*", default=list(COMMANDS), help="commands to run, in order")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for step in args.steps:
        if step not in COMMANDS:
            raise SystemExit("unknown step %r; choose from %s" % (step, ", ".join(COMMANDS)))
        print("== " + step)
        print(json.dumps(COMMANDS[step](cfg), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
