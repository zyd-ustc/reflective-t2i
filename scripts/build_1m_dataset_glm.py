"""1,000,000-pair (100w) multi-process + 64-NPU compositional edit dataset builder.

Combines two concurrent engines into one unified LinearManifest-compatible output root:
  1. Stream A (synth64)   : 64 Qwen-Image-2.1 vLLM-Omni servers across 4 x 16 Ascend 910C nodes
                            + Batch-4 GLM-5.3-flash observe/verify (~350k newly synthesized count/spatial pairs).
  2. Stream B (curate_mp) : Multi-process Batch-4 stitched GLM-5.3-flash semantic curation over
                            L_full_490k.json (T3/T5/T8/T9) + L_edit_final.jsonl (14.94M pool) (~650k+ pairs).

Key production guarantees:
  - Persistent HTTP/1.1 Keep-Alive connection pools (requests.Session) + exponential backoff.
  - Batch-4 vision packing with automatic 1-by-1 fallback so malformed batch JSON never drops valid candidates.
  - Immediate unlink() of rejected generated images so failed edits consume 0 disk bytes.
  - Deterministic namespaced family -> split hashing so 0 train/val/test family collisions can ever occur.
  - Append-only per-shard JSONL checkpoints for instant crash-free resume.
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qwen_latent_cot.count_spatial.candidates import BENCHMARKS, MAX_RATIO  # noqa: E402
from qwen_latent_cot.count_spatial.editor import EditClient  # noqa: E402
from qwen_latent_cot.count_spatial.glm import GlmClient, resolve_glm_endpoint, resolve_glm_key  # noqa: E402
from qwen_latent_cot.count_spatial.manifest import PROTOCOL, ROLE, TASK_CODE  # noqa: E402
from qwen_latent_cot.count_spatial.observe import OBSERVE_PROMPT, POSITIONS, check_observation, suitable  # noqa: E402
from qwen_latent_cot.count_spatial.plan import FAMILIES, choose  # noqa: E402
from qwen_latent_cot.count_spatial.verify import (  # noqa: E402
    check as verify_pair,
    coarse_edit_region,
    edit_semantics,
    instruction as write_repair_instruction,
)
from scripts.curate_edit_pool_glm import (  # noqa: E402
    ALLOWED_490K_TASKS,
    GLM_VERIFY_PROMPT,
    TASK_BUCKETS,
    check_glm_verdict,
    classify_candidate_bucket,
)

DEFAULT_OUT = "/data/zyd_workspace/isochronic_data_20261107/prod_1m_glm53_20261118"
DEFAULT_INDEX_CSV = "/data/250010019/unimoe_r3/qwen_image_vllm_20261019/dataset/index.csv"
DEFAULT_L490K = "/data/zyd_workspace/isochronic_data_20261107/linear_nocaption/L_full_490k.json"
DEFAULT_UNIMOE = "/data/zyd_workspace/isochronic_data_20261107/unimoe_edit/L_edit_final.jsonl"
DEFAULT_FLEET_ENDPOINTS = "/data/zyd_workspace/isochronic_data_20261107/prod_1m_glm53_20261118/fleet64/endpoints.json"

CURATE_BATCH_PROMPT = """You are verifying {n} image-editing training pairs (indexed 0 to {last}).
Each image shows BEFORE (source) on the LEFT and AFTER (target) on the RIGHT, separated by a white bar.
Do NOT judge pixel-level background preservation. Judge ONLY:
1. task_type: one of "count_add", "object_add", "count_remove", "object_remove", "spatial_move", "attr_change", "other".
2. semantic_pass (true/false): true ONLY if AFTER clearly accomplishes the instruction relative to BEFORE (exact requested count, discrete physical object, spatial move, or attribute), names a real physical object (not OCR text/logos or broken nouns like "times"), and both sides are clear scenes.

Instructions by index:
{lines}

Return compact JSON only with exactly {n} items:
{{"items": [{{"idx": 0, "task_type": "<label>", "semantic_pass": <true|false>, "reason": "<under 12 words>"}}, ...]}}"""

OBSERVE_BATCH_PROMPT = """Inspect {n} independent images (indexed 0 to {last}) so an image editor can later change a count or spatial relation in each.
Report ONLY what you clearly see in each image. Focus on prominent foreground/mid-ground objects (countable groups of 2..8 instances and clear left/right/above/below relations).
POSITION must be one of: {positions}.

Return compact JSON only with exactly {n} observations in order:
{{"items": [
  {{"idx": 0, "objects": [{{"name": "<singular noun>", "count": <int>, "positions": ["<POS>"], "countable": <bool>}}], "layout": "<short sentence>", "count_candidates": [{{"object": "<name>", "count": <int>, "reason": "<short>"}}], "spatial_relations": [{{"subject": "<name>", "relation": "left of|right of|above|below|on|inside", "reference": "<name>", "clear": <bool>}}], "quality": "<good|acceptable|poor>", "quality_notes": ""}}
]}}"""

SINGLE_STRIP_VERIFY_PROMPT = """You are inspecting ONE side-by-side image strip of two pictures of the same scene.
The LEFT half is TARGET (picture 1). The RIGHT half is WRONG (picture 2), separated by a thin white vertical bar.
Report ONLY what you clearly see in LEFT ("target") and RIGHT ("wrong") independently.
POSITION must be one of: {positions}.
{vocab_hint}
Return compact JSON only:
{{"target": {{"objects": [{{"name": "<singular noun>", "count": <int>, "positions": ["<POS>"], "countable": <bool>}}], "layout": "<short>", "spatial_relations": [{{"subject": "<name>", "relation": "left of|right of|above|below|on|inside", "reference": "<name>", "clear": <bool>}}], "quality": "<good|acceptable|poor>", "quality_notes": ""}}, "wrong": {{"objects": [{{"name": "<singular noun>", "count": <int>, "positions": ["<POS>"], "countable": <bool>}}], "layout": "<short>", "spatial_relations": [{{"subject": "<name>", "relation": "left of|right of|above|below|on|inside", "reference": "<name>", "clear": <bool>}}], "quality": "<good|acceptable|poor>", "quality_notes": ""}}}}"""


def _pick_glm_key(index=0):
    raw = os.environ.get("METTO_GLM_KEYS") or resolve_glm_key()
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys[index % len(keys)]


def stable_split(family_key):
    h = int(hashlib.sha1(str(family_key).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < 2 else ("val" if h < 4 else "train")


def encode_single_image(path, max_side=512, quality=85):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_side_by_side(left_path, right_path, side=384, quality=85):
    with Image.open(left_path) as li, Image.open(right_path) as ri:
        l_rgb = li.convert("RGB")
        r_rgb = ri.convert("RGB")
        l_rgb.thumbnail((side, side), Image.Resampling.BILINEAR)
        r_rgb.thumbnail((side, side), Image.Resampling.BILINEAR)
        w = l_rgb.width + r_rgb.width + 6
        h = max(l_rgb.height, r_rgb.height)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        canvas.paste(l_rgb, (0, 0))
        canvas.paste(r_rgb, (l_rgb.width + 6, 0))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _scan_unimoe_chunk(args):
    path, start_byte, end_byte = args
    matched = []
    with open(path, "rb") as fh:
        if start_byte > 0:
            fh.seek(start_byte)
            fh.readline()
        while fh.tell() < end_byte:
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("split") not in ("train", "val", "test"):
                continue
            ins = " ".join(str(r.get("edit_instruction") or "").split())
            bucket = classify_candidate_bucket(ins, r.get("task"))
            if not bucket:
                continue
            ds = str(r.get("dataset") or "unimoe")
            raw_fam = str(r.get("source_family") or r.get("uid"))
            ns_fam = f"{ds}:{raw_fam}"
            matched.append(
                dict(
                    uid=f"{ds}:{r['uid']}" if not str(r["uid"]).startswith(ds + ":") else str(r["uid"]),
                    split=stable_split(ns_fam),
                    source_image=r["source_image"],
                    target_image=r["target_image"],
                    edit_instruction=ins,
                    source_family=ns_fam,
                    prompt_family=ns_fam,
                    dataset=ds,
                    candidate_bucket=bucket,
                )
            )
    return matched


# ------------------------------ STEP 0: PREPARE ------------------------------

def cmd_prepare(args):
    out = Path(args.out).resolve()
    synth_root = out / "synth64"
    curate_root = out / "curate_mp"
    synth_root.mkdir(parents=True, exist_ok=True)
    curate_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 1. Prepare 64 shards for Stream A (Qwen-Image-2.1 64-NPU synthesis)
    synth_done = (synth_root / "PREPARE_DONE.json")
    if not synth_done.exists():
        t0 = time.time()
        usable = []
        with open(args.index_csv, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["benchmark"] not in BENCHMARKS:
                    continue
                w, h = int(row["width"]), int(row["height"])
                if max(w, h) / max(1, min(w, h)) > MAX_RATIO:
                    continue
                if not row["image"].lower().endswith(".png"):
                    continue
                uid = row["uid"]
                ns_fam = f"t2i_synth:{uid}"
                usable.append(
                    dict(
                        uid=uid,
                        benchmark=row["benchmark"],
                        wh_ratio=row["wh_ratio"],
                        width=w,
                        height=h,
                        image=row["image"],
                        source_family=ns_fam,
                        prompt_family=ns_fam,
                        split=stable_split(ns_fam),
                    )
                )
        rng.shuffle(usable)
        if args.synth_candidate_limit and args.synth_candidate_limit < len(usable):
            usable = usable[: args.synth_candidate_limit]
        n_shards = args.synth_shards
        for s in range(n_shards):
            sdir = synth_root / "shards" / f"shard_{s:03d}"
            sdir.mkdir(parents=True, exist_ok=True)
            slice_rows = usable[s::n_shards]
            (sdir / "candidates.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slice_rows),
                encoding="utf-8",
            )
        meta_a = dict(usable=len(usable), shards=n_shards, seconds=round(time.time() - t0, 2))
        synth_done.write_text(json.dumps(meta_a, indent=2) + "\n")
        print("Stream A prepared:", json.dumps(meta_a))
    else:
        print("Stream A already prepared:", synth_done.read_text().strip())

    # 2. Prepare 32 shards for Stream B (Existing 15.43M pool multi-process curation)
    curate_done = curate_root / "PREPARE_DONE.json"
    if not curate_done.exists():
        t0 = time.time()
        by_bucket = {b: [] for b in TASK_BUCKETS}
        tpl_counts = Counter()

        if args.l490k and Path(args.l490k).exists():
            recs = json.loads(Path(args.l490k).read_text(encoding="utf-8"))["records"]
            rng.shuffle(recs)
            for r in recs:
                if r.get("split") not in ("train", "val", "test") or str(r.get("task")) not in ALLOWED_490K_TASKS:
                    continue
                ins = " ".join(str(r.get("edit_instruction") or "").split())
                bucket = classify_candidate_bucket(ins, r.get("task"))
                if not bucket:
                    continue
                norm_tpl = re.sub(r"\d+", "#", ins.lower())
                if tpl_counts[norm_tpl] >= args.max_per_template:
                    continue
                tpl_counts[norm_tpl] += 1
                ds = "v2_" + str(r.get("task"))
                ns_fam = f"l490k:{r.get('source_family') or r['uid']}"
                by_bucket[bucket].append(
                    dict(
                        uid=f"l490k:{r['uid']}",
                        split=stable_split(ns_fam),
                        source_image=r["source_image"],
                        target_image=r["target_image"],
                        edit_instruction=ins,
                        source_family=ns_fam,
                        prompt_family=ns_fam,
                        dataset=ds,
                        candidate_bucket=bucket,
                    )
                )

        if args.unimoe and Path(args.unimoe).exists():
            fsize = Path(args.unimoe).stat().st_size
            n_scan = max(1, min(24, os.cpu_count() or 8))
            step = max(1, fsize // n_scan)
            ranges = [(args.unimoe, i * step, fsize if i == n_scan - 1 else (i + 1) * step) for i in range(n_scan)]
            with ProcessPoolExecutor(max_workers=n_scan) as ex:
                for chunk_rows in ex.map(_scan_unimoe_chunk, ranges):
                    rng.shuffle(chunk_rows)
                    for row in chunk_rows:
                        ins = row["edit_instruction"]
                        norm_tpl = re.sub(r"\d+", "#", ins.lower())
                        if tpl_counts[norm_tpl] >= args.max_per_template:
                            continue
                        tpl_counts[norm_tpl] += 1
                        by_bucket[row["candidate_bucket"]].append(row)

        per_bucket_cap = max(1, args.curate_candidate_limit // len(TASK_BUCKETS))
        selected = []
        overflow = []
        bucket_raw = {}
        for b in TASK_BUCKETS:
            rows = by_bucket[b]
            rng.shuffle(rows)
            bucket_raw[b] = len(rows)
            selected.extend(rows[:per_bucket_cap])
            overflow.extend(rows[per_bucket_cap:])

        if len(selected) < args.curate_candidate_limit and overflow:
            rng.shuffle(overflow)
            need = args.curate_candidate_limit - len(selected)
            selected.extend(overflow[:need])

        rng.shuffle(selected)
        seen_uids = set()
        deduped = []
        for r in selected:
            if r["uid"] not in seen_uids:
                seen_uids.add(r["uid"])
                deduped.append(r)

        c_shards = args.curate_shards
        for s in range(c_shards):
            sdir = curate_root / "shards" / f"shard_{s:03d}"
            sdir.mkdir(parents=True, exist_ok=True)
            slice_rows = deduped[s::c_shards]
            (sdir / "candidates.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in slice_rows),
                encoding="utf-8",
            )
        meta_b = dict(
            raw_by_bucket=bucket_raw,
            selected=len(deduped),
            shards=c_shards,
            seconds=round(time.time() - t0, 2),
        )
        curate_done.write_text(json.dumps(meta_b, indent=2) + "\n")
        print("Stream B prepared:", json.dumps(meta_b))
    else:
        print("Stream B already prepared:", curate_done.read_text().strip())


# ------------------------ STREAM B: CURATE SHARD WORKER -----------------------

def verify_curate_batch4(glm_client, batch):
    """Batch-4 stitched verification with automatic 1-by-1 fallback on any error."""
    valid_rows = []
    strips = []
    missing = []
    for r in batch:
        if not (os.path.exists(r["source_image"]) and os.path.exists(r["target_image"])):
            missing.append((r, None, "missing source or target image on disk"))
            continue
        try:
            strips.append(encode_side_by_side(r["source_image"], r["target_image"], side=384, quality=85))
            valid_rows.append(r)
        except Exception as exc:
            missing.append((r, None, f"image decode error: {exc}"))

    if not valid_rows:
        return missing

    if len(valid_rows) >= 2:
        lines = "\n".join(f"{idx}: {r['edit_instruction']}" for idx, r in enumerate(valid_rows))
        prompt = CURATE_BATCH_PROMPT.format(n=len(valid_rows), last=len(valid_rows) - 1, lines=lines)
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in strips
        ]

        def _check_batch(val):
            items = val.get("items")
            if not isinstance(items, list) or len(items) != len(valid_rows):
                raise ValueError(f"expected {len(valid_rows)} items, got {type(items)}")
            for it in items:
                check_glm_verdict(it)

        try:
            reply = glm_client.call(
                [{"role": "user", "content": content}],
                max_tokens=4096,
                check=_check_batch,
                stage="curate_batch4",
            )
            by_idx = {int(it.get("idx", i)): it for i, it in enumerate(reply["items"])}
            out = list(missing)
            for i, r in enumerate(valid_rows):
                out.append((r, by_idx.get(i, reply["items"][i]), None))
            return out
        except Exception:
            pass  # Fall through to 1-by-1 rescue

    out = list(missing)
    for r in valid_rows:
        try:
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": GLM_VERIFY_PROMPT.format(instruction=r["edit_instruction"])},
                        {"type": "image_url", "image_url": {"url": encode_single_image(r["source_image"], 512)}},
                        {"type": "image_url", "image_url": {"url": encode_single_image(r["target_image"], 512)}},
                    ],
                }
            ]
            val = glm_client.call(msgs, max_tokens=4096, check=check_glm_verdict, stage="curate_single", uid=r["uid"])
            out.append((r, val, None))
        except Exception as exc:
            out.append((r, None, str(exc)))
    return out


def cmd_run_curate_shard(args):
    out_root = Path(args.out).resolve()
    stop_flag = out_root / "STOP.flag"
    sdir = out_root / "curate_mp" / "shards" / f"shard_{args.shard_id:03d}"
    cand_p = sdir / "candidates.jsonl"
    ver_p = sdir / "manifest.jsonl"
    rej_p = sdir / "rejects.jsonl"

    cands = [json.loads(l) for l in cand_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    for p in (ver_p, rej_p):
        if p.exists():
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            done.add(json.loads(line)["uid"])
                        except Exception:
                            pass
    todo = [r for r in cands if r["uid"] not in done]
    if not todo:
        print(f"curate shard {args.shard_id:03d}: already complete ({len(done)}/{len(cands)})")
        return

    glm = GlmClient(
        api_key=_pick_glm_key(args.shard_id),
        base_url=resolve_glm_endpoint(),
        model="glm-5.3-flash",
        workers=args.threads,
        retries=3,
        timeout=120,
        max_tokens=4096,
    )

    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    wave_size = max(1, args.threads)
    with ver_p.open("a", encoding="utf-8") as vf, rej_p.open("a", encoding="utf-8") as rf:
        for w0 in range(0, len(batches), wave_size):
            if stop_flag.exists():
                break
            wave = batches[w0 : w0 + wave_size]
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                results = list(pool.map(lambda b: verify_curate_batch4(glm, b), wave))
            for group in results:
                for row, val, err in group:
                    if err or val is None:
                        rf.write(json.dumps({"uid": row["uid"], "reason": str(err)}) + "\n")
                        continue
                    if val.get("semantic_pass") and val.get("task_type") in TASK_BUCKETS:
                        out_row = {
                            **row,
                            "verified_task_type": val["task_type"],
                            "glm_reason": val.get("reason", ""),
                            "semantic_approved": True,
                            "condition_protocol": PROTOCOL,
                            "role": ROLE,
                        }
                        vf.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    else:
                        rf.write(json.dumps({"uid": row["uid"], "verdict": val}) + "\n")
            vf.flush()
            rf.flush()


# ------------------------- STREAM A: SYNTH SHARD WORKER ----------------------

def observe_batch(glm_client, rows):
    jobs = []
    out = []
    by_uid = {}
    for r in rows:
        if not os.path.exists(r["image"]):
            out.append((r, None, "anchor image missing"))
            continue
        try:
            u = encode_single_image(r["image"], max_side=512, quality=85)
        except Exception as exc:
            out.append((r, None, f"decode failed: {exc}"))
            continue
        by_uid[r["uid"]] = r
        msgs = [{"role": "user", "content": [{"type": "text", "text": OBSERVE_PROMPT}, {"type": "image_url", "image_url": {"url": u}}]}]
        jobs.append((r["uid"], msgs, {"check": check_observation, "uid": r["uid"], "max_tokens": 4096}))
    if jobs:
        res = glm_client.call_many(jobs)
        for uid, (obs, err) in res.items():
            out.append((by_uid[uid], obs, err))
    return out


def verify_synth_batch(glm_client, items):
    """items: list of (row, plan, wrong_path, target_path). Uses 1 stitched strip per pair (zero double-call waste)."""
    if not items:
        return []
    from scripts.count_spatial_edits import check_pair

    jobs = []
    by_uid = {}
    out = []
    for item in items:
        row, plan, wrong_p, target_p = item
        try:
            strip_u = encode_side_by_side(target_p, wrong_p, side=448, quality=88)
        except Exception as exc:
            out.append((item, None, str(exc)))
            continue
        nouns = [plan.get("object")] if plan.get("task") == "count" else [plan.get("subject"), plan.get("reference")]
        nouns = [n for n in nouns if n]
        vocab_hint = (
            f"Vocabulary alignment: when visible, use exact singular noun(s) {', '.join(repr(n) for n in nouns)} rather than synonyms."
            if nouns
            else ""
        )
        prompt = SINGLE_STRIP_VERIFY_PROMPT.format(positions=", ".join(POSITIONS), vocab_hint=vocab_hint)
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": strip_u}}]}]
        by_uid[row["uid"]] = item
        jobs.append((row["uid"], msgs, {"check": check_pair, "uid": row["uid"], "max_tokens": 4096}))
    if jobs:
        res = glm_client.call_many(jobs)
        for uid, (val, err) in res.items():
            out.append((by_uid[uid], val, err))
    return out


def _cleanup_files(*paths):
    for p in paths:
        try:
            if p.is_symlink() or p.exists():
                p.unlink()
        except Exception:
            pass


def cmd_run_synth_shard(args):
    out_root = Path(args.out).resolve()
    stop_flag = out_root / "STOP.flag"
    shard_ids = [int(x) for x in str(args.shard_ids).split(",") if x.strip()] if getattr(args, "shard_ids", None) else [args.shard_id]
    endpoints = [u.strip() for u in str(args.endpoint).split(",") if u.strip()]

    glm = GlmClient(
        api_key=_pick_glm_key(shard_ids[0]),
        base_url=resolve_glm_endpoint(),
        model="glm-5.3-flash",
        workers=1,
        retries=3,
        timeout=120,
        max_tokens=4096,
    )
    editors = [
        EditClient(u, "Qwen/Qwen-Image-2.1", steps=args.steps, cfg_scale=1.0, timeout=300, workers=1)
        for u in endpoints
    ]
    ed_turn = 0

    for sid in shard_ids:
        if stop_flag.exists():
            return
        sdir = out_root / "synth64" / "shards" / f"shard_{sid:03d}"
        img_dir = sdir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        cand_p = sdir / "candidates.jsonl"
        ver_p = sdir / "manifest.jsonl"
        rej_p = sdir / "rejects.jsonl"

        cands = [json.loads(l) for l in cand_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        done = set()
        for p in (ver_p, rej_p):
            if p.exists():
                with p.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            try:
                                done.add(json.loads(line)["uid"])
                            except Exception:
                                pass
        todo = [r for r in cands if r["uid"] not in done]
        if not todo:
            continue

        plan_idx = len(done)
        with ver_p.open("a", encoding="utf-8") as vf, rej_p.open("a", encoding="utf-8") as rf:
            for b0 in range(0, len(todo), 1):
                if stop_flag.exists():
                    return
                chunk = todo[b0 : b0 + 1]
                obs_results = observe_batch(glm, chunk)
                planned_items = []
                for row, obs, err in obs_results:
                    uid = row["uid"]
                    if err or obs is None:
                        rf.write(json.dumps({"uid": uid, "step": "observe", "reason": str(err)}) + "\n")
                        continue
                    if not suitable(obs):
                        rf.write(json.dumps({"uid": uid, "step": "observe", "reason": "not suitable"}) + "\n")
                        continue
                    plan = choose(row["image"], obs, plan_idx)
                    plan_idx += 1
                    if plan is None:
                        rf.write(json.dumps({"uid": uid, "step": "plan", "reason": "no plan"}) + "\n")
                        continue
                    planned_items.append((row, plan))

                edited_items = []
                for row, plan in planned_items:
                    uid = row["uid"]
                    editor = editors[ed_turn % len(editors)]
                    ed_turn += 1
                    try:
                        out_edit = editor.edit(Path(row["image"]), plan["breakage_instruction"], seed=42 + (plan_idx % 997))
                    except Exception as exc:
                        rf.write(json.dumps({"uid": uid, "step": "edit", "reason": str(exc)}) + "\n")
                        continue
                    if plan["edit_family"] == "count_increase":
                        gen_path = img_dir / f"{uid}_target.jpg"
                        orig_path = img_dir / f"{uid}_wrong.png"
                        wrong_p, target_p = orig_path, gen_path
                    else:
                        gen_path = img_dir / f"{uid}_wrong.jpg"
                        orig_path = img_dir / f"{uid}_target.png"
                        wrong_p, target_p = gen_path, orig_path
                    _cleanup_files(gen_path, orig_path)
                    out_edit["image"].save(gen_path, format="JPEG", quality=95, subsampling=0)
                    orig_path.symlink_to(Path(row["image"]).resolve())
                    edited_items.append((row, plan, wrong_p, target_p))

                ver_results = verify_synth_batch(glm, edited_items)
                for (row, plan, wrong_p, target_p), pair_val, err in ver_results:
                    uid = row["uid"]
                    if err or pair_val is None:
                        _cleanup_files(wrong_p, target_p)
                        rf.write(json.dumps({"uid": uid, "step": "verify", "reason": str(err)}) + "\n")
                        continue
                    verdict = verify_pair(pair_val, plan, strict_preservation=False)
                    repair_text = write_repair_instruction(verdict)
                    if not verdict["passed"] or not repair_text:
                        _cleanup_files(wrong_p, target_p)
                        rf.write(json.dumps({"uid": uid, "step": "verify", "reason": "; ".join(verdict["reasons"])}) + "\n")
                        continue
                    family = plan["edit_family"]
                    rec = dict(
                        uid=f"synth:{uid}",
                        split=row["split"],
                        source_image=str(wrong_p),
                        target_image=str(target_p),
                        edit_instruction=repair_text,
                        tail_condition=pair_val.get("target", {}).get("layout", ""),
                        task=TASK_CODE[family],
                        role=ROLE,
                        semantic_task=family,
                        count_direction=plan["count_direction"],
                        source_family=row["source_family"],
                        prompt_family=row["prompt_family"],
                        semantic_approved=True,
                        condition_protocol=PROTOCOL,
                        tail_source="declared",
                        benchmark=row.get("benchmark"),
                        edit_semantics=edit_semantics(verdict),
                        edit_region=coarse_edit_region(verdict),
                    )
                    vf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                vf.flush()
                rf.flush()


# ----------------------- LAUNCH, MONITOR & STATUS ----------------------------

def collect_status(out_dir, target_total=400_000):
    out = Path(out_dir).resolve()
    started_p = out / "STARTED_AT.txt"
    now = time.time()
    started = float(started_p.read_text().strip()) if started_p.exists() else now
    elapsed_h = max(1e-4, (now - started) / 3600.0)

    def _count_lines(glob_pat):
        total = 0
        for p in out.glob(glob_pat):
            with p.open("rb") as fh:
                total += sum(1 for _ in fh)
        return total

    synth_ver = _count_lines("synth64/shards/shard_*/manifest.jsonl")
    synth_rej = _count_lines("synth64/shards/shard_*/rejects.jsonl")
    curate_ver = _count_lines("curate_mp/shards/shard_*/manifest.jsonl")
    curate_rej = _count_lines("curate_mp/shards/shard_*/rejects.jsonl")
    total_ver = synth_ver + curate_ver
    rate_per_h = total_ver / elapsed_h if (now - started) >= 30 else 0.0
    rem = max(0, target_total - total_ver)
    eta_h = round(rem / rate_per_h, 2) if rate_per_h > 0 else None

    return dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        elapsed_hours=round(elapsed_h, 3),
        target_total=target_total,
        verified_total=total_ver,
        progress_pct=round(100.0 * total_ver / max(1, target_total), 3),
        synth64=dict(
            verified=synth_ver,
            rejected=synth_rej,
            pass_rate=round(synth_ver / max(1, synth_ver + synth_rej), 4),
            per_hour=round(synth_ver / elapsed_h, 1) if (now - started) >= 30 else None,
        ),
        curate_mp=dict(
            verified=curate_ver,
            rejected=curate_rej,
            pass_rate=round(curate_ver / max(1, curate_ver + curate_rej), 4),
            per_hour=round(curate_ver / elapsed_h, 1) if (now - started) >= 30 else None,
        ),
        combined_verified_per_hour=round(rate_per_h, 1) if rate_per_h > 0 else None,
        eta_hours_to_target=eta_h,
    )


def export_consolidated_manifests(out_dir, shard_size=10000):
    from qwen_latent_cot.count_spatial.manifest import validate as validate_manifest

    out = Path(out_dir).resolve()
    final_dir = out / "final_manifests"
    final_dir.mkdir(parents=True, exist_ok=True)
    for old_f in final_dir.glob("manifest_*.jsonl"):
        old_f.unlink()
    seen = set()
    rows = []
    skipped_missing = 0
    for pat in ("synth64/shards/shard_*/manifest.jsonl", "curate_mp/shards/shard_*/manifest.jsonl"):
        for p in sorted(out.glob(pat)):
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    uid = r.get("uid")
                    if not uid or uid in seen:
                        continue
                    src_p, tgt_p = r.get("source_image"), r.get("target_image")
                    if not (src_p and tgt_p and os.path.exists(src_p) and os.path.exists(tgt_p)):
                        skipped_missing += 1
                        continue
                    seen.add(uid)
                    rows.append(r)

    shards = []
    for i in range(0, len(rows), shard_size):
        sp = final_dir / f"manifest_{len(shards):03d}.jsonl"
        sp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows[i : i + shard_size]), encoding="utf-8")
        shards.append(sp)

    cfg_path = REPO / "configs/u15_isochronic_state/isochronic_manifold.json"
    report = validate_manifest(shards, cfg_path) if shards else {"pairs": 0, "train": 0, "splits": {}}
    by_task = Counter(r.get("verified_task_type") or r.get("semantic_task") or "other" for r in rows)
    summary = dict(
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        skipped_missing_images=skipped_missing,
        by_task_type=dict(by_task),
        shards=[str(s) for s in shards],
        **report,
    )
    (out / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def cmd_status(args):
    st = collect_status(args.out, args.target)
    (Path(args.out) / "STATUS.json").write_text(json.dumps(st, indent=2) + "\n")
    print(json.dumps(st, indent=2, ensure_ascii=False))


def cmd_export(args):
    print(json.dumps(export_consolidated_manifests(args.out), indent=2, ensure_ascii=False))


def cmd_monitor(args):
    out = Path(args.out).resolve()
    while True:
        st = collect_status(out, args.target)
        (out / "STATUS.json").write_text(json.dumps(st, indent=2) + "\n")
        if st["verified_total"] >= args.target:
            (out / "STOP.flag").write_text(st["timestamp"] + "\n")
            time.sleep(8)
            export_consolidated_manifests(out)
            fleet_script = REPO / "scripts/launch_editor_fleet64.py"
            if fleet_script.exists() and os.environ.get("VC_WORKER_HOSTS"):
                subprocess.run([sys.executable, str(fleet_script), "--action", "stop"], check=False)
            break
        time.sleep(60)


def cmd_launch_workers(args):
    out = Path(args.out).resolve()
    (out / "STOP.flag").unlink(missing_ok=True)
    log_dir = out / "worker_logs"
    pid_dir = out / "worker_pids"
    log_dir.mkdir(parents=True, exist_ok=True)
    pid_dir.mkdir(parents=True, exist_ok=True)
    if not (out / "STARTED_AT.txt").exists():
        (out / "STARTED_AT.txt").write_text(str(time.time()) + "\n")

    endpoints = json.loads(Path(args.endpoints).read_text(encoding="utf-8"))
    py = sys.executable
    script = str(Path(__file__).resolve())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    launched_synth = 0
    n_sw = max(0, int(args.synth_workers))
    for w in range(n_sw):
        assigned_shards = [str(s) for s in range(w, args.synth_shards, n_sw)]
        assigned_eps = [endpoints[s % len(endpoints)]["url"] for s in range(w, len(endpoints), n_sw)]
        pid_f = pid_dir / f"synth_w{w:03d}.pid"
        if pid_f.exists():
            old_pid = pid_f.read_text().strip()
            if old_pid.isdigit() and Path(f"/proc/{old_pid}").exists():
                continue
        log_f = (log_dir / f"synth_w{w:03d}.log").open("a")
        proc = subprocess.Popen(
            [
                py,
                script,
                "run-synth-shard",
                "--out",
                str(out),
                "--shard-id",
                str(w),
                "--shard-ids",
                ",".join(assigned_shards),
                "--endpoint",
                ",".join(assigned_eps),
                "--steps",
                str(args.steps),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        pid_f.write_text(str(proc.pid) + "\n")
        launched_synth += 1

    launched_curate = 0
    for s in range(args.curate_shards):
        pid_f = pid_dir / f"curate_{s:03d}.pid"
        if pid_f.exists():
            old_pid = pid_f.read_text().strip()
            if old_pid.isdigit() and Path(f"/proc/{old_pid}").exists():
                continue
        log_f = (log_dir / f"curate_{s:03d}.log").open("a")
        proc = subprocess.Popen(
            [
                py,
                script,
                "run-curate-shard",
                "--out",
                str(out),
                "--shard-id",
                str(s),
                "--threads",
                str(args.curate_threads),
                "--batch-size",
                str(args.curate_batch_size),
            ],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        pid_f.write_text(str(proc.pid) + "\n")
        launched_curate += 1

    mon_pid_f = pid_dir / "monitor.pid"
    if mon_pid_f.exists():
        old_mp = mon_pid_f.read_text().strip()
        if old_mp.isdigit():
            subprocess.run(["kill", old_mp], check=False)
    mon_log = (log_dir / "monitor.log").open("a")
    mon = subprocess.Popen(
        [py, script, "monitor", "--out", str(out), "--target", str(args.target)],
        stdout=mon_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    mon_pid_f.write_text(str(mon.pid) + "\n")

    print(
        json.dumps(
            dict(
                launched_synth_workers=launched_synth,
                launched_curate_workers=launched_curate,
                target=args.target,
                out=str(out),
            ),
            indent=2,
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare")
    pp.add_argument("--out", default=DEFAULT_OUT)
    pp.add_argument("--index-csv", default=DEFAULT_INDEX_CSV)
    pp.add_argument("--l490k", default=DEFAULT_L490K)
    pp.add_argument("--unimoe", default=DEFAULT_UNIMOE)
    pp.add_argument("--synth-shards", type=int, default=64)
    pp.add_argument("--synth-candidate-limit", type=int, default=672455)
    pp.add_argument("--curate-shards", type=int, default=24)
    pp.add_argument("--curate-candidate-limit", type=int, default=900000)
    pp.add_argument("--max-per-template", type=int, default=600)
    pp.add_argument("--seed", type=int, default=2026)

    pc = sub.add_parser("run-curate-shard")
    pc.add_argument("--out", default=DEFAULT_OUT)
    pc.add_argument("--shard-id", type=int, required=True)
    pc.add_argument("--threads", type=int, default=1)
    pc.add_argument("--batch-size", type=int, default=6)

    ps = sub.add_parser("run-synth-shard")
    ps.add_argument("--out", default=DEFAULT_OUT)
    ps.add_argument("--shard-id", type=int, required=True)
    ps.add_argument("--shard-ids", default="")
    ps.add_argument("--endpoint", required=True)
    ps.add_argument("--steps", type=int, default=24)

    pl = sub.add_parser("launch-workers")
    pl.add_argument("--out", default=DEFAULT_OUT)
    pl.add_argument("--endpoints", default=DEFAULT_FLEET_ENDPOINTS)
    pl.add_argument("--synth-shards", type=int, default=64)
    pl.add_argument("--synth-workers", type=int, default=16)
    pl.add_argument("--curate-shards", type=int, default=24)
    pl.add_argument("--curate-threads", type=int, default=1)
    pl.add_argument("--curate-batch-size", type=int, default=6)
    pl.add_argument("--steps", type=int, default=24)
    pl.add_argument("--target", type=int, default=400000)

    pst = sub.add_parser("status")
    pst.add_argument("--out", default=DEFAULT_OUT)
    pst.add_argument("--target", type=int, default=400000)

    pex = sub.add_parser("export")
    pex.add_argument("--out", default=DEFAULT_OUT)

    pm = sub.add_parser("monitor")
    pm.add_argument("--out", default=DEFAULT_OUT)
    pm.add_argument("--target", type=int, default=400000)

    args = p.parse_args()
    {
        "prepare": cmd_prepare,
        "run-curate-shard": cmd_run_curate_shard,
        "run-synth-shard": cmd_run_synth_shard,
        "launch-workers": cmd_launch_workers,
        "status": cmd_status,
        "export": cmd_export,
        "monitor": cmd_monitor,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
