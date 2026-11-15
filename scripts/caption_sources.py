"""Caption the source images so a T2I start can be built honestly.

The held-out test wanted a "deployment start": a T2I trajectory walked to the
read time. The L cache has no scene description, so the first version passed an
empty prompt, which made the model draw unrelated pictures — that column was
answering nothing.

This asks a vision model for a short, factual description of each source image
and stores it per uid. The description must not name the edit: it says what the
frame contains, which is what a deployer would type.

    caption_sources.py --manifest ... --split val --out captions_val.jsonl
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

BASE = "https://metto-api.com"
PATH = "/v1/chat/completions"
MODEL = "glm-5.3-flash"

STYLE = (
    "Describe this photograph in one comma-separated list: the main subjects and how many "
    "of each, their colours and materials, the setting, and the camera framing. Do not "
    "describe any change, do not use words like 'current', 'should' or 'instead', and do "
    "not give commands. Between 12 and 35 words. Output the description only."
)


def _block(path, max_edge):
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, float(max_edge) / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))),
                                 Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def caption(session, key, path, max_edge, retries=3):
    payload = dict(model=MODEL, stream=False, temperature=0.2, max_tokens=120,
                   messages=[dict(role="user", content=[
                       dict(type="text", text=STYLE),
                       dict(type="image_url", image_url=dict(url=_block(path, max_edge)))])])
    last = None
    for attempt in range(retries):
        try:
            response = session.post(BASE + PATH,
                                    headers={"Authorization": "Bearer " + key,
                                             "Content-Type": "application/json"},
                                    json=payload, timeout=(15, 120))
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(json.dumps(data["error"], ensure_ascii=False))
            text = " ".join((data["choices"][0]["message"].get("content") or "").split())
            if not text:
                raise RuntimeError("empty caption")
            return text, None
        except Exception as error:
            last = f"{type(error).__name__}: {str(error)[:140]}"
            time.sleep(1.5 * (attempt + 1))
    return None, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--image-max-edge", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--key-env", default="METTO_KEY")
    args = parser.parse_args()

    key = os.environ.get(args.key_env, "").strip()
    if not key:
        parser.error("set " + args.key_env)

    records = [r for r in json.loads(args.manifest.read_text())["records"]
               if r["split"] == args.split]
    # The same balanced pick the edit test uses, so captions line up with cases.
    def family(text):
        text = (text or "").lower()
        if text.startswith(("add ", "remove ")):
            return "count"
        if text.startswith(("rearrange", "swap", "move")):
            return "spatial"
        if text.startswith("replace"):
            return "replace"
        if text.startswith("change"):
            return "attribute"
        return "other"

    buckets = {}
    for record in records:
        buckets.setdefault(family(record.get("edit_instruction")), []).append(record)
    rng = random.Random(args.seed)
    for values in buckets.values():
        values.sort(key=lambda r: r["uid"])
        rng.shuffle(values)
    chosen = []
    while len(chosen) < args.limit and any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name] and len(chosen) < args.limit:
                chosen.append(buckets[name].pop())

    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("caption"):
                    done.add(row["uid"])
    print(f"selected {len(chosen)} cases, {len(done)} already captioned", flush=True)

    lock = threading.Lock()
    local = threading.local()
    counts = dict(ok=0, failed=0, skipped=0)

    def work(record):
        uid = record["uid"]
        if uid in done:
            with lock:
                counts["skipped"] += 1
            return
        session = getattr(local, "session", None)
        if session is None:
            session = local.session = requests.Session()
        text, error = caption(session, key, record["source_image"], args.image_max_edge)
        with lock:
            with args.out.open("a") as handle:
                handle.write(json.dumps(dict(uid=uid, split=args.split, caption=text,
                                             error=error), ensure_ascii=False) + "\n")
            counts["ok" if text else "failed"] += 1
            total = counts["ok"] + counts["failed"]
            if total % 50 == 0:
                print(json.dumps(counts), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, chosen))
    print("final:", json.dumps(counts), flush=True)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
