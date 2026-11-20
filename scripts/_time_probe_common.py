"""Shared tensor statistics, paired bootstrap CIs, and comparison strip rendering."""

import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


def cos_sim(a, b):
    fa = a.detach().float().reshape(1, -1)
    fb = b.detach().float().reshape(1, -1)
    return float(F.cosine_similarity(fa, fb, dim=1))


def lowpass(x, factor=8):
    return F.avg_pool2d(x.float(), factor, factor)


def split_freq(x, factor=8):
    low_small = lowpass(x, factor)
    low = F.interpolate(low_small, size=x.shape[-2:], mode="bilinear", align_corners=False)
    high = x.float() - low
    return low, high


def grad_mag(x):
    dx = F.pad(x.float()[:, :, :, 1:] - x.float()[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(x.float()[:, :, 1:, :] - x.float()[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.pow(2) + dy.pow(2) + 1e-12).mean(dim=1, keepdim=True)


def masked_mean(val, mask):
    denom = mask.float().sum().clamp_min(1.0)
    if val.shape[1] != mask.shape[1]:
        denom = denom * (val.shape[1] / mask.shape[1])
    return float((val.float() * mask.float()).sum() / denom)


def mean_or_none(vals):
    clean = [float(v) for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def paired_bootstrap_ci(vals, n_boot=1000, seed=17):
    """Return (mean, ci_low_95, ci_high_95, n) for numeric values."""
    clean = [float(v) for v in vals if v is not None]
    n = len(clean)
    if n == 0:
        return None, None, None, 0
    mean_val = sum(clean) / n
    if n == 1:
        return mean_val, mean_val, mean_val, 1
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += clean[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[max(0, int(0.025 * (n_boot - 1)))]
    hi = boots[min(n_boot - 1, int(0.975 * (n_boot - 1)))]
    return mean_val, lo, hi, n


def tensor_to_pil(tensor):
    t = tensor.detach().float().clamp(-1.0, 1.0)[0]
    arr = ((t.permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(pil_img, device="cpu"):
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device) / 127.5 - 1.0


def save_labeled_strip(panels, out_path, tile_size=384, quality=90):
    """Save horizontal labeled comparison strip (JPEG/PNG) for visual inspection."""
    n = len(panels)
    banner_h = 34
    gap = 6
    width = n * tile_size + (n - 1) * gap
    height = banner_h + tile_size
    canvas = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (img_tensor, label) in enumerate(panels):
        x0 = i * (tile_size + gap)
        pil_im = tensor_to_pil(img_tensor).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(pil_im, (x0, banner_h))
        draw.text((x0 + 8, 9), str(label)[:48], fill=(245, 245, 245))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=quality)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
