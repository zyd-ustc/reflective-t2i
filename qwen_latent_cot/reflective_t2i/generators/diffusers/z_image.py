"""Z-Image Base (6B single-stream S3-DiT) GeneratorAdapter with strict ImageTokenMaskedLoRA.

Reuses the battle-tested NPU-native ``zcorr.zimage_core`` implementation when available
(e.g., ``/data/zyd_workspace/zimage-isochronic-correction/src``), and enforces strict
single-stream token isolation so LoRA updates apply ONLY to image latent tokens while
text caption tokens and ``context_refiner`` remain 100% frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from PIL import Image
import torch
from torch import nn

from ...core.contracts import ConditionBundle
from ...core.noise_geometry import LinearFlowGeometry
from ...editors.latent_lora import LatentLoRAConfig, LatentLoRACorrector, TokenMaskContext
from ..base import GeneratorAdapter


DEFAULT_ZIMAGE_CORE_SRC = "/data/zyd_workspace/zimage-isochronic-correction/src"


def _import_zimage_core(core_src: str | None = None):
    candidates = [core_src, os.environ.get("ZIMAGE_CORE_SRC"), DEFAULT_ZIMAGE_CORE_SRC]
    for path_str in candidates:
        if path_str and Path(path_str).exists() and str(path_str) not in sys.path:
            sys.path.insert(0, str(path_str))
    from zcorr.zimage_core import (  # type: ignore[import-not-found]
        decode_latents,
        encode_latents,
        load_zimage_components,
    )
    from zcorr.zimage_core.constants import (  # type: ignore[import-not-found]
        BASE_IMAGE_SEQ_LEN,
        BASE_SHIFT,
        MAX_IMAGE_SEQ_LEN,
        MAX_SHIFT,
        SEQ_MULTI_OF,
    )
    return {
        "BASE_IMAGE_SEQ_LEN": BASE_IMAGE_SEQ_LEN,
        "BASE_SHIFT": BASE_SHIFT,
        "MAX_IMAGE_SEQ_LEN": MAX_IMAGE_SEQ_LEN,
        "MAX_SHIFT": MAX_SHIFT,
        "SEQ_MULTI_OF": SEQ_MULTI_OF,
        "decode_latents": decode_latents,
        "encode_latents": encode_latents,
        "load_zimage_components": load_zimage_components,
    }


@dataclass
class ZImageConfig:
    model_path: str = "/data/zyd_workspace/models/Z-Image"
    zimage_core_src: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    max_sequence_length: int = 512
    dtype: str = "bfloat16"
    t_eps: float = 0.05
    lora_rank: int = 16
    lora_alpha: float | None = None
    layer_scope: str | Sequence[int] = "middle_third"
    adapt_noise_refiner: bool = False
    target_suffixes: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")


class ZImageAdapter(GeneratorAdapter):
    """Frozen Z-Image Base S3-DiT with token-masked image-only LoRA corrector."""

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        scheduler: Any,
        config: ZImageConfig,
        *,
        core_helpers: dict[str, Any] | None = None,
        attach_corrector: bool = True,
    ) -> None:
        self.transformer = transformer.eval().requires_grad_(False)
        self.vae = vae.eval().requires_grad_(False)
        self.text_encoder = text_encoder.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.config = config
        self._core = core_helpers or _import_zimage_core(config.zimage_core_src)
        self._device = next(transformer.parameters()).device
        self._model_dtype = next(transformer.parameters()).dtype
        self._geometry = LinearFlowGeometry(
            prediction_type="v_to_clean",
            noise_scale=1.0,
            t_eps=float(config.t_eps),
        )
        self._token_mask_ctx = TokenMaskContext()
        self._active_image_token_len: int | None = None
        self._hook_handle = None

        # Pre-hook on layers[0] reads exact unified sequence length (B, L_unified, D)
        # and masks LoRA strictly to indices [0 : x_len) (image tokens only).
        if hasattr(self.transformer, "layers") and len(self.transformer.layers) > 0:
            self._hook_handle = self.transformer.layers[0].register_forward_pre_hook(
                self._set_single_stream_image_mask
            )

        self.corrector: LatentLoRACorrector | None = None
        if attach_corrector:
            include_prefixes = ("layers.", "noise_refiner.") if config.adapt_noise_refiner else ("layers.",)
            lora_cfg = LatentLoRAConfig(
                adapter_name="corrector",
                rank=int(config.lora_rank),
                alpha=config.lora_alpha,
                layer_scope=config.layer_scope,
                block_prefix="layers",
                include_prefixes=include_prefixes,
                exclude_substrings=("context_refiner",),
                target_suffixes=config.target_suffixes,
                masked_prefixes=("layers.",),
            )
            self.corrector = LatentLoRACorrector(
                self.transformer,
                lora_cfg,
                num_blocks=len(self.transformer.layers),
                token_mask_ctx=self._token_mask_ctx,
            )

    @classmethod
    def from_pretrained(cls, config: ZImageConfig, *, device: str | torch.device = "npu:0", attach_corrector: bool = True) -> "ZImageAdapter":
        core = _import_zimage_core(config.zimage_core_src)
        dev = torch.device(device)
        dtype = torch.bfloat16 if config.dtype == "bfloat16" and dev.type != "cpu" else torch.float32
        components = core["load_zimage_components"](config.model_path, device=dev, dtype=dtype)
        return cls(
            transformer=components["transformer"],
            vae=components["vae"],
            text_encoder=components["text_encoder"],
            tokenizer=components["tokenizer"],
            scheduler=components["scheduler"],
            config=config,
            core_helpers=core,
            attach_corrector=attach_corrector,
        )

    def _set_single_stream_image_mask(self, _module, args):
        if not args or self._active_image_token_len is None:
            return
        unified = args[0]
        if not isinstance(unified, torch.Tensor) or unified.ndim != 3:
            return
        bsz, seq_len, _ = unified.shape
        img_len = min(int(self._active_image_token_len), seq_len)
        mask = torch.zeros((bsz, seq_len, 1), device=unified.device, dtype=torch.bool)
        mask[:, :img_len, 0] = True
        self._token_mask_ctx._mask = mask

    @property
    def geometry(self) -> LinearFlowGeometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self._device

    def _latent_hw(self) -> tuple[int, int]:
        scale_factor = getattr(self.vae, "vae_scale_factor", 8)
        return self.config.height // scale_factor, self.config.width // scale_factor

    def _image_seq_len(self, h: int, w: int) -> int:
        p = int(getattr(self.transformer, "all_patch_size", (2,))[0])
        seq_mult = int(self._core["SEQ_MULTI_OF"])
        raw = (h // p) * (w // p)
        pad = (-raw) % seq_mult
        return raw + pad

    def progress_grid(self) -> list[float]:
        lh, lw = self._latent_hw()
        p = int(getattr(self.transformer, "all_patch_size", (2,))[0])
        img_seq = (lh // p) * (lw // p)
        base_seq = self._core["BASE_IMAGE_SEQ_LEN"]
        max_seq = self._core["MAX_IMAGE_SEQ_LEN"]
        base_shift = self._core["BASE_SHIFT"]
        max_shift = self._core["MAX_SHIFT"]
        m = (max_shift - base_shift) / (max_seq - base_seq)
        mu = float(img_seq * m + (base_shift - m * base_seq))

        steps = int(self.config.num_inference_steps)
        sigmas = [1.0 - i / steps for i in range(steps + 1)]
        use_dyn = bool(getattr(self.scheduler.config, "use_dynamic_shifting", False))
        shift = float(getattr(self.scheduler.config, "shift", 3.0))
        shifted = []
        for s in sigmas:
            if s <= 0.0:
                shifted.append(0.0)
            elif s >= 1.0:
                shifted.append(1.0)
            elif use_dyn:
                shifted.append(math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0)))
            else:
                shifted.append(shift * s / (1.0 + (shift - 1.0) * s))
        return [round(1.0 - s, 7) for s in shifted]

    def sample_initial_noise(self, *, seed: int, batch_size: int = 1) -> torch.Tensor:
        lh, lw = self._latent_hw()
        channels = int(getattr(self.transformer, "in_channels", 16))
        gen = torch.Generator(device=self._device).manual_seed(int(seed))
        return self._geometry.sample_noise(
            (batch_size, channels, lh, lw),
            device=self._device,
            dtype=torch.float32,
            generator=gen,
        )

    @torch.no_grad()
    def encode_prompt(self, text: str | None) -> ConditionBundle:
        prompt_str = "" if text is None else str(text)
        messages = [{"role": "user", "content": prompt_str}]
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded = self.tokenizer(
            [formatted],
            padding="max_length",
            max_length=int(self.config.max_sequence_length),
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self._device)
        attn_mask = encoded.attention_mask.to(self._device).bool()
        out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        feat = out.hidden_states[-2][0][attn_mask[0]].to(dtype=self._model_dtype)
        return ConditionBundle(raw_text=text, payload=[feat], mode="z_image")

    @torch.no_grad()
    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor:
        return self._core["encode_latents"](self.vae, rgb_minus1_1.to(self._device)).float()

    @torch.no_grad()
    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image:
        rgb = self._core["decode_latents"](self.vae, clean_tensor.to(self._device))
        arr = ((rgb[0].float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())

    def predict_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        cond: ConditionBundle,
        *,
        use_corrector: bool = False,
    ) -> torch.Tensor:
        if self.corrector is not None:
            self.corrector.set_active(bool(use_corrector))
        bsz, _, lh, lw = z.shape
        t_clean = torch.as_tensor(p, device=self._device, dtype=torch.float32)
        if t_clean.ndim == 0:
            t_clean = t_clean.expand(bsz)
        x_list = [z[i].to(device=self._device, dtype=self._model_dtype).unsqueeze(1) for i in range(bsz)]
        cap_list = cond.payload if len(cond.payload) == bsz else [cond.payload[0]] * bsz
        cap_list = [c.to(device=self._device, dtype=self._model_dtype) for c in cap_list]

        self._active_image_token_len = self._image_seq_len(lh, lw)
        self._token_mask_ctx._mask = None
        try:
            pred_list = self.transformer(x_list, t_clean, cap_list)[0]
        finally:
            self._active_image_token_len = None
            self._token_mask_ctx._mask = None
            if self.corrector is not None:
                self.corrector.set_active(False)

        v_pred = torch.stack([item.squeeze(1) for item in pred_list], dim=0).float()
        return self._geometry.to_clean(z.float(), t_clean, v_pred)
