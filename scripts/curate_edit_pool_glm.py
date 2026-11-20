"""Task-type balanced curation & GLM-5.3-flash semantic verification for existing edit pools.

Focuses strictly on semantic task-type coverage and edit execution (no pixel-preservation rejection):
  1. select   : sample balanced candidates across 6 compositional task types from
                L_full_490k.json (T3/T5/T8/T9 only) and L_edit_final.jsonl, capping repeated templates.
  2. verify   : verify semantic task execution and task_type with Metto glm-5.3-flash.
  3. manifest : export LinearManifest-validated training/val JSONL.
"""

import argparse
import base64
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qwen_latent_cot.count_spatial.glm import GlmClient, resolve_glm_endpoint, resolve_glm_key  # noqa: E402
from qwen_latent_cot.count_spatial.manifest import validate as validate_manifest  # noqa: E402

TASK_BUCKETS = (
    "count_add",
    "object_add",
    "count_remove",
    "object_remove",
    "spatial_move",
    "attr_change",
)

BANNED_KEYWORDS = re.compile(
    r"\b(?:makeup|lipstick|eyeshadow|eyeliner|older|younger|wrinkles|beard|mustache|"
    r"earring|earrings|necklace|sunglasses|baseball\s+cap|tattoo|hairstyle|hair\s+colou?r|"
    r"rainbow|galaxy\s+background|replace\s+the\s+background|change\s+the\s+background|"
    r"season\s+to|winter|autumn|snow\s+on|fog|misty|restyle|ink\s+wash|oil\s+painting|"
    r"watercolor|anime|cartoon\s+style|camera\s+angle|low-angle|watermark|logo|lettering|"
    r"caption|subtitle|word|words|text|translate\s+the\s+visible\s+text)\b|"
    r"(?:['\"][^'\"]{1,40}['\"])|"
    r"\b(?:add|remove)\s+\d+\s+(?:times?|things?|items?|pieces?|parts?|elements?|areas?|sides?)\b",
    flags=re.I,
)
ALLOWED_490K_TASKS = {"T3", "T5", "T8", "T9"}
NUM_RE = re.compile(r"\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\b", flags=re.I)

GLM_VERIFY_PROMPT = """You are verifying an image-editing training pair for a compositional Text-to-Image correction dataset.
Image 1 is the BEFORE (source) image. Image 2 is the AFTER (target) image.
Instruction: "{instruction}"

Do NOT judge pixel-level background preservation or minor style shifts. Focus ONLY on semantic task coverage and whether the requested semantic action actually occurred between Image 1 and Image 2:
1. Classify task_type into one of:
   - "count_add": increases the count of an existing object type (e.g., 2 -> 3 cups) or adds multiple countable instances
   - "object_add": adds 1 new discrete standalone object to the scene (not weather/background/makeup)
   - "count_remove": reduces the count of a multi-instance group while leaving >= 1 remaining (e.g., 5 -> 3)
   - "object_remove": removes a discrete object completely from the scene
   - "spatial_move": moves, swaps, or rearranges object left/right/above/below/next-to positions
   - "attr_change": changes a discrete object's colour, material, or replaces one discrete object with another
   - "other": global style/weather/background filter, text OCR, portrait makeup/aging, or compound multi-step edit
2. Check semantic_pass (true/false):
   - true ONLY if Image 2 clearly accomplishes the instruction relative to Image 1 (exact requested count, discrete physical object, spatial relation, or attribute), the instruction names a real visible physical object (not a broken noun like "times" or OCR text/letters/logos), and both images are clear scenes.
   - false if the edit did not happen, achieved a wrong count, edited text/signs/logos, used a nonsensical target noun, or belongs to "other".

Return compact JSON only:
{{"task_type": "<one of the 7 labels>", "semantic_pass": <true or false>, "reason": "<under 20 words>"}}"""


def classify_candidate_bucket(instruction, task=None):
    text = " ".join(str(instruction or "").split()).strip()
    if not text or BANNED_KEYWORDS.search(text):
        return None
    low = text.lower()
    if task == "T8":
        if low.startswith("add "):
            return "count_add"
        if low.startswith("remove "):
            return "count_remove"
    if low.startswith("add "):
        if "instead of the current" in low or (NUM_RE.search(low) and not re.match(r"^add\s+(?:a|an|1|one)\b", low)):
            return "count_add"
        return "object_add"
    if low.startswith("remove "):
        if "instead of the current" in low or (NUM_RE.search(low) and not re.match(r"^remove\s+(?:a|an|the|1|one)\b", low)):
            return "count_remove"
        return "object_remove"
    if task == "T9" or low.startswith(("move ", "rearrange ", "swap ", "relocate ", "reposition ")):
        return "spatial_move"
    if task in ("T3", "T5") or low.startswith(("change ", "replace ", "turn ")):
        return "attr_change"
    return None


def _encode_thumb(path, max_side=768):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def check_glm_verdict(val):
    if "task_type" not in val or "semantic_pass" not in val:
        raise ValueError("missing task_type or semantic_pass")
    if not isinstance(val["semantic_pass"], bool):
        raise ValueError("semantic_pass must be boolean")


def cmd_select(args):
    rng = random.Random(args.seed)
    pools = defaultdict(list)
    tpl_counts = Counter()

    if args.l490k and Path(args.l490k).exists():
        recs = json.loads(Path(args.l490k).read_text())["records"]
        rng.shuffle(recs)
        for r in recs:
            if r.get("split") not in ("train", "val") or str(r.get("task")) not in ALLOWED_490K_TASKS:
                continue
            ins = " ".join(str(r.get("edit_instruction") or "").split())
            bucket = classify_candidate_bucket(ins, r.get("task"))
            if not bucket:
                continue
            norm_tpl = re.sub(r"\d+", "#", ins.lower())
            if tpl_counts[norm_tpl] >= args.max_per_template:
                continue
            tpl_counts[norm_tpl] += 1
            pools[bucket].append(
                dict(
                    uid=r["uid"],
                    split=r["split"],
                    source_image=r["source_image"],
                    target_image=r["target_image"],
                    edit_instruction=ins,
                    source_family=r["source_family"],
                    prompt_family=r["prompt_family"],
                    dataset="v2_" + str(r.get("task")),
                    candidate_bucket=bucket,
                )
            )

    if args.unimoe and Path(args.unimoe).exists():
        allowed_ds = {"ImgEdit", "AnyEdit", "SEED-Data-Edit", "MagicBrush", "X2Edit-Dataset"}
        with Path(args.unimoe).open() as fh:
            for idx, line in enumerate(fh):
                if args.unimoe_scan_limit and idx >= args.unimoe_scan_limit:
                    break
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") not in ("train", "val") or r.get("dataset") not in allowed_ds:
                    continue
                ins = " ".join(str(r.get("edit_instruction") or "").split())
                bucket = classify_candidate_bucket(ins)
                if not bucket:
                    continue
                norm_tpl = re.sub(r"\d+", "#", ins.lower())
                if tpl_counts[norm_tpl] >= args.max_per_template:
                    continue
                if len(pools[bucket]) < args.per_bucket * 4:
                    tpl_counts[norm_tpl] += 1
                    pools[bucket].append(
                        dict(
                            uid=r["uid"],
                            split=r["split"],
                            source_image=r["source_image"],
                            target_image=r["target_image"],
                            edit_instruction=ins,
                            source_family=r["source_family"],
                            prompt_family=r["prompt_family"],
                            dataset=r["dataset"],
                            candidate_bucket=bucket,
                        )
                    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    summary = {}
    seen_fams_split = {}
    for bucket in TASK_BUCKETS:
        cands = pools[bucket]
        rng.shuffle(cands)
        kept = []
        for row in cands:
            if len(kept) >= args.per_bucket:
                break
            fam = row["source_family"]
            prev = seen_fams_split.get(fam)
            if prev is not None and prev != row["split"]:
                continue
            if not (Path(row["source_image"]).exists() and Path(row["target_image"]).exists()):
                continue
            seen_fams_split[fam] = row["split"]
            kept.append(row)
        selected.extend(kept)
        summary[bucket] = len(kept)

    out_path = out_dir / "candidates.jsonl"
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    print(json.dumps({"written": len(selected), "by_bucket": summary, "path": str(out_path)}, indent=2))


def cmd_verify(args):
    out_dir = Path(args.out).resolve()
    cand_path = out_dir / "candidates.jsonl"
    ver_path = out_dir / "verified.jsonl"
    rej_path = out_dir / "rejects.jsonl"

    cands = [json.loads(l) for l in cand_path.read_text().splitlines() if l.strip()]
    done = set()
    if ver_path.exists():
        done |= {json.loads(l)["uid"] for l in ver_path.read_text().splitlines() if l.strip()}
    if rej_path.exists():
        done |= {json.loads(l)["uid"] for l in rej_path.read_text().splitlines() if l.strip()}
    todo = [r for r in cands if r["uid"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    client = GlmClient(
        api_key=resolve_glm_key(),
        base_url=resolve_glm_endpoint(),
        model="glm-5.3-flash",
        workers=args.workers,
        retries=3,
        timeout=180,
        max_tokens=2048,
    )

    jobs = []
    by_uid = {r["uid"]: r for r in todo}
    for r in todo:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": GLM_VERIFY_PROMPT.format(instruction=r["edit_instruction"])},
                    {"type": "image_url", "image_url": {"url": _encode_thumb(r["source_image"])}},
                    {"type": "image_url", "image_url": {"url": _encode_thumb(r["target_image"])}},
                ],
            }
        ]
        jobs.append((r["uid"], msgs, {"check": check_glm_verdict, "stage": "curate_verify", "uid": r["uid"]}))

    results = client.call_many(jobs) if jobs else {}
    passed = 0
    rejected = 0
    with ver_path.open("a") as vf, rej_path.open("a") as rf:
        for uid, (val, err) in results.items():
            row = by_uid[uid]
            if err or val is None:
                rf.write(json.dumps({"uid": uid, "reason": str(err)}) + "\n")
                rejected += 1
                continue
            if val["semantic_pass"] and val["task_type"] in TASK_BUCKETS:
                out_row = {
                    **row,
                    "verified_task_type": val["task_type"],
                    "glm_reason": val.get("reason", ""),
                    "semantic_approved": True,
                    "condition_protocol": "semantic_feedback_only_v1",
                    "role": "L_edit",
                }
                vf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                passed += 1
            else:
                rf.write(json.dumps({"uid": uid, "verdict": val}) + "\n")
                rejected += 1
    print(json.dumps({"checked": len(results), "passed": passed, "rejected": rejected}, indent=2))


def cmd_manifest(args):
    out_dir = Path(args.out).resolve()
    ver_path = out_dir / "verified.jsonl"
    rows = [json.loads(l) for l in ver_path.read_text().splitlines() if l.strip()]
    manifest_path = out_dir / "curated_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    cfg_path = REPO / "configs/u15_isochronic_state/isochronic_manifold.json"
    report = validate_manifest([manifest_path], cfg_path)
    by_type = Counter(r["verified_task_type"] for r in rows)
    print(json.dumps({"manifest": str(manifest_path), **report, "by_task_type": dict(by_type)}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("select")
    ps.add_argument("--l490k", default="/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json")
    ps.add_argument("--unimoe", default="/data/zyd_workspace/isochronic_data_20261107/unimoe_edit/L_edit_final.jsonl")
    ps.add_argument("--unimoe-scan-limit", type=int, default=400000)
    ps.add_argument("--per-bucket", type=int, default=50)
    ps.add_argument("--max-per-template", type=int, default=15)
    ps.add_argument("--seed", type=int, default=2026)
    ps.add_argument("--out", required=True)

    pv = sub.add_parser("verify")
    pv.add_argument("--out", required=True)
    pv.add_argument("--workers", type=int, default=16)
    pv.add_argument("--limit", type=int, default=0)

    pm = sub.add_parser("manifest")
    pm.add_argument("--out", required=True)

    args = p.parse_args()
    {"select": cmd_select, "verify": cmd_verify, "manifest": cmd_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
