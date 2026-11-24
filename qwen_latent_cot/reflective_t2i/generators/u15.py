"""SenseNova-U1.5 MoT adapter bridging legacy u15_target_correction into reflective_t2i_v1."""

from __future__ import annotations

from typing import Any

from PIL import Image
import torch

from ..core.contracts import ConditionBundle, StateCorrector
from ..core.noise_geometry import LinearFlowGeometry
from .base import GeneratorAdapter


class _U15CorrectorWrapper(StateCorrector):
    """Thin StateCorrector adapter over existing IsochronicManifoldEditor."""

    def __init__(self, editor: Any) -> None:
        self.editor = editor
        self._active = False

    def set_active(self, active: bool) -> None:
        self._active = bool(active)

    @property
    def active(self) -> bool:
        return self._active

    def trainable_parameters(self) -> dict[str, torch.nn.Parameter]:
        from ...u15_target_correction.checkpoint import canonical_parameters

        return canonical_parameters(self.editor)

    def state_dict_trainable(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.trainable_parameters().items()}

    def load_state_dict_trainable(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        params = self.trainable_parameters()
        missing = sorted(set(params.keys()) - set(state_dict.keys()))
        unexpected = sorted(set(state_dict.keys()) - set(params.keys()))
        if strict and (missing or unexpected):
            raise KeyError(f"U15 corrector mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        with torch.no_grad():
            for k, p in params.items():
                if k in state_dict:
                    p.copy_(state_dict[k].to(device=p.device, dtype=p.dtype))


class U15Adapter(GeneratorAdapter):
    """Wraps existing SenseNova-U1.5 Runtime + IsochronicManifoldEditor under GeneratorAdapter."""

    def __init__(self, rt: Any, editor: Any, cfg: dict[str, Any]) -> None:
        self.rt = rt
        self.editor = editor
        self.cfg = dict(cfg)
        self._geometry = LinearFlowGeometry(
            prediction_type="clean_x",
            noise_scale=float(rt.noise_scale()),
            t_eps=float(cfg.get("t_eps", 0.05)),
        )
        self.corrector = _U15CorrectorWrapper(editor)

    @property
    def geometry(self) -> LinearFlowGeometry:
        return self._geometry

    @property
    def device(self) -> torch.device:
        return self.rt.device

    def progress_grid(self) -> list[float]:
        return [float(t) for t in self.rt.times()]

    def sample_initial_noise(self, *, seed: int, batch_size: int = 1) -> torch.Tensor:
        gen = torch.Generator(device=self.rt.device).manual_seed(int(seed))
        return self._geometry.sample_noise(
            (batch_size, 3, int(self.rt.height), int(self.rt.width)),
            device=self.rt.device,
            dtype=torch.float32,
            generator=gen,
        )

    @torch.no_grad()
    def encode_prompt(self, text: str | None) -> ConditionBundle:
        pref = self.rt.prefix(text)
        return ConditionBundle(raw_text=text, payload=(pref.cache, pref.indexes, None), mode="u15")

    @torch.no_grad()
    def encode_clean(self, rgb_minus1_1: torch.Tensor) -> torch.Tensor:
        return rgb_minus1_1.to(device=self.rt.device, dtype=torch.float32)

    @torch.no_grad()
    def decode_preview(self, clean_tensor: torch.Tensor) -> Image.Image:
        arr = ((clean_tensor[0].detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        return Image.fromarray(arr.permute(1, 2, 0).cpu().numpy())

    def predict_clean(
        self,
        z: torch.Tensor,
        p: float | torch.Tensor,
        cond: ConditionBundle,
        *,
        use_corrector: bool = False,
    ) -> torch.Tensor:
        tau = float(p.mean().item()) if isinstance(p, torch.Tensor) else float(p)
        if use_corrector:
            return self.editor.predict_clean(z, tau, cond.payload)
        return self.editor.native_clean(z, tau, cond.payload)
