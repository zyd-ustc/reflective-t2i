"""Stable Diffusion 3.5 Large (8B dual-stream MMDiT) GeneratorAdapter.

Strictly isolates LoRA updates to the MMDiT image stream (``attn.to_q/k/v/to_out.0``,
excluding text-stream ``add_*`` and ``ff_context``), and solves CLIP bag-of-words
imperative blindness by routing declarative ``target_rewrite`` to CLIP-L/G while routing
``Original | Edit`` to T5-XXL under ``native_prompt_plus_feedback``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image
import torch
from torch import nn

from ...core.contracts import ConditionBundle, ConditionMode
from ...core.noise_geometry import LinearFlowGeometry
from ...editors.latent_lora import LatentLoRAConfig, LatentLoRACorrector
from ..base import GeneratorAdapter


@dataclass
class SD35Config:
    model_id: str = "stabilityai/stable-diffusion-3.5-large"
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 40
    max_sequence_length: int = 256
    dtype: str = "bfloat16"
    t_eps: float = 0.05
    lora_rank: int = 16
    lora_alpha: float | None = None
    layer_scope: str | Sequence[int] = "middle_third"
    include_ff: bool = False


class SD35Adapter(GeneratorAdapter):
    """Frozen Diffusers SD3.5-Large MMDiT with strict image-stream-only LoRA corrector."""

    def __init__(
        self,
        pipe: Any,
        config: SD35Config,
        *,
        attach_corrector: bool = True,
    ) -> None:
        self.pipe = pipe
        self.transformer: nn.Module = pipe.transformer.eval().requires_grad_(False)
        self.vae: nn.Module = pipe.vae.eval().requires_grad_(False)
        self.scheduler = pipe.scheduler
        self.config = config
        self._device = next(self.transformer.parameters()).device
        self._model_dtype = next(self.transformer.parameters()).dtype
        # Diffusers FlowMatch predicts v = eps - x_0 -> prediction_type = "v_to_noise"
        self._geometry = LinearFlowGeometry(
            prediction_type="v_to_noise",
            noise_scale=1.0,
            t_eps=float(config.t_eps),
        )

        self.corrector: LatentLoRACorrector | None = None
        if attach_corrector:
            targets = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"]
            if config.include_ff:
                targets.extend(["ff.net.0.proj", "ff.net.2"])
            lora_cfg = LatentLoRAConfig(
                adapter_name="corrector",
                rank=int(config.lora_rank),
                alpha=config.lora_alpha,
                layer_scope=config.layer_scope,
                block_prefix="transformer_blocks",
                include_prefixes=("transformer_blocks.",),
                exclude_substrings=("add_q_proj", "add_k_proj", "add_v_proj", "to_add_out", "ff_context"),
                target_suffixes=tuple(targets),
            )
            num_blocks = len(getattr(self.transformer, "transformer_blocks", ()))
            self.corrector = LatentLoRACorrector(self.transformer, lora_cfg, num_blocks=num_blocks)

    @classmethod
    def from_pretrained(
        cls,
        config: SD35Config,
        *,
        device: str | torch.device = "npu:0",
        attach_corrector: bool = True,
    ) -> "SD35Adapter":
        from diffusers import StableDiffusion3Pipeline  # type: ignore[import-not-found]

        dev = torch.device(device)
        dtype = torch.bfloat16 if config.dtype == "bfloat16" and dev.type != "cpu" else torch.float32
        pipe = StableDiffusion3Pipeline.from_pretrained(config.model_id, torch_dtype=dtype)
        pipe = pipe.to(dev)
        return cls(pipe=pipe, config=config, attach_corrector=attach_corrector)

    @property
    def geometry(self) -> LinearFlowGeometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self._device

    def progress_grid(self) -> list[float]:
        self.scheduler.set_timesteps(int(self.config.num_inference_steps), device=self._device)
        sigmas = [float(s) for s in self.scheduler.sigmas.detach().cpu().tolist()]
        # scheduler.sigmas goes 1.0 -> 0.0; normalized progress p = 1.0 - sigma goes 0.0 -> 1.0
        return [round(1.0 - s, 7) for s in sigmas]

    def sample_initial_noise(self, *, seed: int, batch_size: int = 1) -> torch.Tensor:
        scale = int(getattr(self.pipe, "vae_scale_factor", 8))
        ch = int(getattr(self.transformer.config, "in_channels", 16))
        gen = torch.Generator(device=self._device).manual_seed(int(seed))
        return self._geometry.sample_noise(
            (batch_size, ch, self.config.height // scale, self.config.width // scale),
            device=self._device,
            dtype=torch.float32,
            generator=gen,
        )

    @torch.no_grad()
    def _encode_clip_t5(self, clip_text: str, t5_text: str) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, _, pooled_embeds, _ = self.pipe.encode_prompt(
            prompt=clip_text,
            prompt_2=clip_text,
            prompt_3=t5_text,
            device=self._device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            max_sequence_length=int(self.config.max_sequence_length),
        )
        return prompt_embeds.to(dtype=self._model_dtype), pooled_embeds.to(dtype=self._model_dtype)

    @torch.no_grad()
    def encode_prompt(self, text: str | None) -> ConditionBundle:
        s = "" if text is None else str(text)
        embeds, pooled = self._encode_clip_t5(s, s)
        return ConditionBundle(raw_text=text, payload=(embeds, pooled), mode="sd35")

    @torch.no_grad()
    def encode_edit_condition(
        self,
        *,
        feedback: str,
        original_prompt: str | None = None,
        target_rewrite: str | None = None,
        condition_mode: ConditionMode = "native_prompt_plus_feedback",
    ) -> ConditionBundle:
        fb = " ".join(str(feedback or "").split()).strip()
        if not fb:
            return self.encode_prompt(None)
        if condition_mode == "feedback_only" or not original_prompt:
            s = f"Edit request: {fb}"
            embeds, pooled = self._encode_clip_t5(s, s)
            return ConditionBundle(raw_text=s, payload=(embeds, pooled), mode="feedback_only")
        # Route declarative scene description to CLIP-L/G (avoiding bag-of-words "remove X" backfire)
        # and full Original + Edit instruction to T5-XXL.
        clip_text = " ".join(str(target_rewrite or original_prompt).split()).strip()
        t5_text = f"Original: {' '.join(str(original_prompt).split()).strip()} | Edit: {fb}"
        embeds, pooled = self._encode_clip_t5(clip_text, t5_text)
        return ConditionBundle(raw_text=t5_text, payload=(embeds, pooled), mode="native_prompt_plus_feedback")

    @torch.no_grad()
    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor:
        lat = self.vae.encode(rgb_minus1_1.to(device=self._device, dtype=self.vae.dtype)).latent_dist.mode()
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        return ((lat.float() - shift) * scale).float()

    @torch.no_grad()
    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image:
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        scale = float(getattr(self.vae.config, "scaling_factor", 1.0))
        lat = (clean_tensor.float() / scale + shift).to(device=self._device, dtype=self.vae.dtype)
        rgb = self.vae.decode(lat, return_dict=False)[0]
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
        bsz = z.shape[0]
        pt = torch.as_tensor(p, device=self._device, dtype=torch.float32)
        if pt.ndim == 0:
            pt = pt.expand(bsz)
        sigma = (1.0 - pt).clamp(0.0, 1.0)
        timestep = sigma * float(getattr(self.scheduler.config, "num_train_timesteps", 1000))

        embeds, pooled = cond.payload
        if embeds.shape[0] != bsz:
            embeds = embeds.expand(bsz, -1, -1)
            pooled = pooled.expand(bsz, -1)

        try:
            v_sd3 = self.transformer(
                hidden_states=z.to(device=self._device, dtype=self._model_dtype),
                timestep=timestep.to(device=self._device, dtype=self._model_dtype),
                encoder_hidden_states=embeds.to(device=self._device, dtype=self._model_dtype),
                pooled_projections=pooled.to(device=self._device, dtype=self._model_dtype),
                return_dict=False,
            )[0].float()
        finally:
            if self.corrector is not None:
                self.corrector.set_active(False)

        return self._geometry.to_clean(z.float(), pt, v_sd3)
