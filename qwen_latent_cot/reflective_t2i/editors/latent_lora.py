"""Switchable zero-init LoRA with optional per-token masking for single-stream S3-DiT isolation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import re
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..core.contracts import StateCorrector


class TokenMaskContext:
    """Mutable carrier holding the active boolean/float image-token mask `(B, L, 1)` during a forward pass."""

    def __init__(self) -> None:
        self._mask: torch.Tensor | None = None

    @property
    def mask(self) -> torch.Tensor | None:
        return self._mask

    @contextmanager
    def apply(self, mask: torch.Tensor | None):
        prev = self._mask
        if mask is not None and mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        self._mask = mask
        try:
            yield
        finally:
            self._mask = prev


class SwitchableLoRALinear(nn.Module):
    """Zero-init B LoRA wrapper over a frozen ``nn.Linear`` with optional token-level masking.

    When ``active_adapter is None``, forward reduces strictly to ``self.base(x)``.
    When ``token_mask_ctx`` carries a `(B, L, 1)` mask matching a 3-D token tensor `x`,
    LoRA updates (and their gradients) are strictly zeroed on non-image tokens (e.g. text
    tokens in single-stream S3-DiT blocks).
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        token_mask_ctx: TokenMaskContext | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self.scalings: dict[str, float] = {}
        self.active_adapter: str | None = None
        self._token_mask_ctx = token_mask_ctx

    def add_adapter(self, name: str, rank: int, alpha: float | None = None) -> None:
        r = int(rank)
        if r < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        device = self.base.weight.device
        a = nn.Parameter(torch.empty((r, self.base.in_features), device=device, dtype=torch.float32))
        b = nn.Parameter(torch.zeros((self.base.out_features, r), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_a[name] = a
        self.lora_b[name] = b
        self.scalings[name] = float(r if alpha is None else alpha) / float(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        name = self.active_adapter
        if name is None or name not in self.lora_a:
            return out
        a = self.lora_a[name].to(dtype=x.dtype)
        b = self.lora_b[name].to(dtype=x.dtype)
        update = F.linear(F.linear(x, a), b) * self.scalings[name]
        if self._token_mask_ctx is not None:
            mask = self._token_mask_ctx.mask
            if mask is not None and update.ndim == 3 and tuple(mask.shape[:2]) == tuple(update.shape[:2]):
                update = update * mask.to(device=update.device, dtype=update.dtype)
        return out + update


def resolve_layer_indices(num_layers: int, scope: str | Sequence[int]) -> set[int]:
    """Resolve ``'all'``, ``'middle_third'``, ``'upper_half'``, or explicit index list."""
    if num_layers <= 0:
        return set()
    if not isinstance(scope, str):
        return {int(i) for i in scope if 0 <= int(i) < num_layers}
    key = scope.strip().lower()
    if key == "all":
        return set(range(num_layers))
    if key == "middle_third":
        lo = num_layers // 3
        hi = max(lo + 1, (2 * num_layers + 2) // 3)
        return set(range(lo, min(hi, num_layers)))
    if key == "upper_half":
        return set(range(num_layers // 2, num_layers))
    raise ValueError(f"Unsupported layer_scope: {scope}")


def _set_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def _extract_block_index(name: str, block_prefix: str) -> int | None:
    m = re.match(rf"^{re.escape(block_prefix)}\.(\d+)\.", name)
    return int(m.group(1)) if m else None


@dataclass
class LatentLoRAConfig:
    adapter_name: str = "corrector"
    rank: int = 16
    alpha: float | None = None
    layer_scope: str | Sequence[int] = "middle_third"
    block_prefix: str = "layers"
    include_prefixes: tuple[str, ...] = ("layers.",)
    exclude_substrings: tuple[str, ...] = ()
    target_suffixes: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    masked_prefixes: tuple[str, ...] = ()


class LatentLoRACorrector(StateCorrector):
    """Attaches switchable (optionally token-masked) LoRA adapters to a frozen DiT."""

    def __init__(
        self,
        backbone: nn.Module,
        config: LatentLoRAConfig,
        *,
        num_blocks: int | None = None,
        token_mask_ctx: TokenMaskContext | None = None,
    ) -> None:
        self.backbone = backbone
        self.config = config
        self.token_mask_ctx = token_mask_ctx or TokenMaskContext()
        self.wrapped: dict[str, SwitchableLoRALinear] = {}

        # Freeze entire backbone before attaching adapters
        self.backbone.eval().requires_grad_(False)

        if num_blocks is None:
            found = [
                idx
                for name, _ in backbone.named_modules()
                if (idx := _extract_block_index(name, config.block_prefix)) is not None
            ]
            num_blocks = (max(found) + 1) if found else 0
        allowed_blocks = resolve_layer_indices(num_blocks, config.layer_scope)

        for name, mod in list(backbone.named_modules()):
            if not isinstance(mod, (nn.Linear, SwitchableLoRALinear)):
                continue
            if config.include_prefixes and not name.startswith(config.include_prefixes):
                continue
            if any(bad in name for bad in config.exclude_substrings):
                continue
            if not any(name.endswith(suf) for suf in config.target_suffixes):
                continue
            blk_idx = _extract_block_index(name, config.block_prefix)
            if blk_idx is not None and allowed_blocks and blk_idx not in allowed_blocks:
                continue

            use_mask = bool(config.masked_prefixes) and name.startswith(config.masked_prefixes)
            if isinstance(mod, SwitchableLoRALinear):
                lora_mod = mod
            else:
                lora_mod = SwitchableLoRALinear(
                    mod,
                    token_mask_ctx=self.token_mask_ctx if use_mask else None,
                )
                _set_submodule(backbone, name, lora_mod)
            lora_mod.add_adapter(config.adapter_name, config.rank, config.alpha)
            self.wrapped[name] = lora_mod

        if not self.wrapped:
            raise ValueError(
                f"LatentLoRACorrector matched 0 modules (prefix={config.include_prefixes}, "
                f"targets={config.target_suffixes}, scope={config.layer_scope})"
            )
        self.set_active(False)

    def set_active(self, active: bool) -> None:
        name = self.config.adapter_name if active else None
        for mod in self.wrapped.values():
            mod.active_adapter = name

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        name = self.config.adapter_name
        params: dict[str, nn.Parameter] = {}
        for mod_name in sorted(self.wrapped.keys()):
            mod = self.wrapped[mod_name]
            params[f"{mod_name}.lora_a.{name}"] = mod.lora_a[name]
            params[f"{mod_name}.lora_b.{name}"] = mod.lora_b[name]
        return params

    def state_dict_trainable(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.trainable_parameters().items()}

    def load_state_dict_trainable(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        params = self.trainable_parameters()
        missing = sorted(set(params.keys()) - set(state_dict.keys()))
        unexpected = sorted(set(state_dict.keys()) - set(params.keys()))
        if strict and (missing or unexpected):
            raise KeyError(f"Corrector state mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        with torch.no_grad():
            for key, param in params.items():
                if key in state_dict:
                    param.copy_(state_dict[key].to(device=param.device, dtype=param.dtype))
