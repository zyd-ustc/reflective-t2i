"""Model-agnostic evaluation primitives for reflective_t2i_v1."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence

from PIL import Image, ImageDraw
import torch

from ..core.contracts import ObserverBackend


def compute_hf_grad(rgb_minus1_1: torch.Tensor) -> float:
    """Mean Sobel/finite-difference high-frequency gradient magnitude on RGB [-1, 1]."""
    x = rgb_minus1_1.detach().float().clamp(-1.0, 1.0)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return float(0.5 * (dx + dy).item())


def compute_oob_frac(raw_tensor: torch.Tensor, bound: float = 1.0) -> float:
    x = raw_tensor.detach().float()
    return float(((x < -bound) | (x > bound)).float().mean().item())


def save_strip(tiles: Sequence[tuple[Image.Image, str]], out_path: str | Path, tile_size: int = 320) -> None:
    if not tiles:
        return
    header = 28
    canvas = Image.new("RGB", (tile_size * len(tiles), tile_size + header), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for idx, (im, label) in enumerate(tiles):
        resized = im.convert("RGB").resize((tile_size, tile_size), Image.Resampling.BICUBIC)
        x0 = idx * tile_size
        canvas.paste(resized, (x0, header))
        draw.text((x0 + 8, 6), label[:42], fill=(240, 240, 240))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


@torch.no_grad()
def probe_read_progress(
    generator_adapter,
    prompts: Sequence[str],
    candidate_progress: Sequence[float] = (0.24, 0.30, 0.36),
    *,
    guidance_scale: float = 4.0,
    cfg_interval: tuple[float, float] = (0.0, 0.80),
    lookahead_steps: int = 4,
    seed: int = 42,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Zero-training calibration probe measuring step fraction and preview fidelity across read_progress."""
    grid = generator_adapter.progress_grid()
    total_steps = max(1, len(grid) - 1)
    null_cond = generator_adapter.encode_prompt(None)

    step_summary = {}
    for p in candidate_progress:
        pf = float(p)
        prefix_steps = sum(1 for v in grid if float(v) < pf - 1e-6)
        mid_steps = sum(1 for v in grid if pf - 1e-6 <= float(v) <= min(0.99, pf + 0.24) + 1e-6)
        step_summary[f"{pf:.2f}"] = {
            "read_progress": pf,
            "prefix_steps": prefix_steps,
            "prefix_frac": round(prefix_steps / total_steps, 3),
            "midband_steps_plus_0.24": mid_steps,
            "tail_steps": total_steps - prefix_steps,
        }

    per_p_l1 = {f"{float(p):.2f}": [] for p in candidate_progress}
    per_p_hf = {f"{float(p):.2f}": [] for p in candidate_progress}
    out_path = Path(out_dir).resolve() if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    import numpy as np

    for idx, prompt in enumerate(prompts):
        cond = generator_adapter.encode_prompt(prompt)
        z0 = generator_adapter.sample_initial_noise(seed=int(seed) + idx)

        # Full native target for preview-to-final alignment measurement
        state_0, _ = generator_adapter.sample_until(
            z0, float(candidate_progress[0]), cond, null_cond, guidance_scale, cfg_interval=cfg_interval, return_clean_at_end=False
        )
        final_latent = generator_adapter.continue_native(state_0, cond, null_cond, guidance_scale, cfg_interval=cfg_interval)
        final_img = generator_adapter.decode_preview(final_latent)
        final_arr = np.asarray(final_img, dtype=np.float32) / 127.5 - 1.0

        tiles = [(final_img, "NATIVE_FINAL")]
        curr_state = state_0
        for p_idx, p in enumerate(candidate_progress):
            pf = float(p)
            if p_idx > 0:
                curr_state, _ = generator_adapter.sample_until(
                    curr_state.tensor,
                    pf,
                    cond,
                    null_cond,
                    guidance_scale,
                    start_p=curr_state.progress,
                    cfg_interval=cfg_interval,
                    return_clean_at_end=False,
                )
            prev_img = generator_adapter.lookahead_preview(
                curr_state, cond, null_cond, guidance_scale, steps=lookahead_steps, cfg_interval=cfg_interval
            )
            prev_arr = np.asarray(prev_img, dtype=np.float32) / 127.5 - 1.0
            l1 = float(np.abs(prev_arr - final_arr).mean())
            hf = compute_hf_grad(torch.from_numpy(prev_arr).permute(2, 0, 1))
            per_p_l1[f"{pf:.2f}"].append(l1)
            per_p_hf[f"{pf:.2f}"].append(hf)
            tiles.append((prev_img, f"PREVIEW p={pf:.2f} (L1={l1:.3f})"))

        if out_path:
            save_strip(tiles, out_path / f"probe_{idx:02d}.jpg")

    for key in step_summary:
        step_summary[key]["mean_preview_to_final_l1"] = round(sum(per_p_l1[key]) / max(len(per_p_l1[key]), 1), 4)
        step_summary[key]["mean_preview_hf_grad"] = round(sum(per_p_hf[key]) / max(len(per_p_hf[key]), 1), 5)

    report = {
        "total_native_steps": total_steps,
        "num_prompts": len(prompts),
        "candidates": step_summary,
    }
    if out_path:
        (out_path / "probe_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


EDIT_PAIR_JUDGE_SYSTEM = (
    "You evaluate whether an edited image successfully followed a requested edit instruction "
    "compared to the original source image, while preserving unrelated regions. "
    "Return ONLY valid JSON: "
    '{"edit_followed": <0 or 1>, "bg_preserved": <0 or 1>, "quality_ok": <0 or 1>, "reason": "<short>"}'
)


def score_edit_pair_vlm(
    observer: ObserverBackend,
    source_img: Image.Image,
    edited_img: Image.Image,
    instruction: str,
) -> dict[str, Any]:
    """Side-by-side VLM judge scoring edit compliance and preservation on an unseen val pair."""
    w, h = 512, 512
    combo = Image.new("RGB", (w * 2 + 12, h + 28), (30, 30, 30))
    draw = ImageDraw.Draw(combo)
    combo.paste(source_img.convert("RGB").resize((w, h), Image.Resampling.BICUBIC), (0, 28))
    combo.paste(edited_img.convert("RGB").resize((w, h), Image.Resampling.BICUBIC), (w + 12, 28))
    draw.text((8, 6), "LEFT: ORIGINAL SOURCE", fill=(240, 240, 240))
    draw.text((w + 20, 6), "RIGHT: EDITED RESULT", fill=(240, 240, 240))

    prompt = (
        f"Edit instruction: {instruction}\n\n"
        "Compare the LEFT (original) and RIGHT (edited) images.\n"
        "1) edit_followed: 1 if the requested edit is clearly and accurately executed (exact count/object/attribute/position), else 0.\n"
        "2) bg_preserved: 1 if unrequested objects and scene layout stay consistent, else 0.\n"
        "3) quality_ok: 1 if the right image has no severe artifacts or ghosts, else 0.\n"
        "Return ONLY the JSON object."
    )
    raw = observer.query_image(combo, prompt, system_prompt=EDIT_PAIR_JUDGE_SYSTEM, max_new_tokens=128, temperature=0.0)
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.I).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(cleaned[start : end + 1])
            return {
                "edit_followed": int(bool(obj.get("edit_followed", 0))),
                "bg_preserved": int(bool(obj.get("bg_preserved", 0))),
                "quality_ok": int(bool(obj.get("quality_ok", 0))),
                "strict_pass": int(
                    bool(obj.get("edit_followed", 0))
                    and bool(obj.get("bg_preserved", 0))
                    and bool(obj.get("quality_ok", 0))
                ),
                "reason": str(obj.get("reason", ""))[:160],
            }
        except Exception:
            pass
    return {"edit_followed": 0, "bg_preserved": 0, "quality_ok": 0, "strict_pass": 0, "reason": f"parse_fail:{cleaned[:80]}"}
