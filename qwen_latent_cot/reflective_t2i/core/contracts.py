"""Core protocols, dataclasses, and abstract interfaces for reflective_t2i_v1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from PIL import Image
import torch


PredictionType = Literal["clean_x", "v_to_clean", "v_to_noise"]
ReflectionOutcome = Literal["satisfied", "edit", "gated_noop"]
EditAction = Literal["none", "add", "remove", "attr", "move", "other"]
AnchorMode = Literal["conditional_native", "empty_native"]
ConditionMode = Literal["feedback_only", "native_prompt_plus_feedback"]


@dataclass
class LatentState:
    """Generator-agnostic intermediate state along normalized progress p in [0, 1]."""

    tensor: torch.Tensor
    progress: float
    valid_mask: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def detach_clone(self) -> "LatentState":
        return LatentState(
            tensor=self.tensor.detach().clone(),
            progress=float(self.progress),
            valid_mask=None if self.valid_mask is None else self.valid_mask.detach().clone(),
            meta=dict(self.meta),
        )


@dataclass
class ConditionBundle:
    """Opaque generator-specific text/multimodal conditioning handle."""

    raw_text: str | None
    payload: Any
    mode: str = "native"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionDecision:
    """Structured output of the decoupled Observer + ReflectionPolicy."""

    outcome: ReflectionOutcome
    action: EditAction = "none"
    instruction: str = ""
    target_rewrite: str = ""
    coarse_region_hint: str = ""
    confidence: float = 1.0
    gated_by: str | None = None
    gated_instruction: str | None = None
    observed_summary: str = ""
    recheck_response: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionResult:
    """Output of IsochronicTransitionV2 state correction."""

    state: LatentState
    clean_target_est: torch.Tensor
    clean_source_est: torch.Tensor
    eps_anchor: torch.Tensor
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full end-to-end reflective T2I inference trace."""

    image: Image.Image
    final_tensor: torch.Tensor
    preview_images: list[Image.Image] = field(default_factory=list)
    decisions: list[ReflectionDecision] = field(default_factory=list)
    transitions: list[TransitionResult] = field(default_factory=list)
    arm: str = "C_reflective"
    telemetry: dict[str, Any] = field(default_factory=dict)


class ObserverBackend(ABC):
    """Decoupled external VLM backend for image observation and structured reflection."""

    @abstractmethod
    def query_image(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        """Run deterministic VLM inference on a single image + prompt."""

    def query_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        """Optional text-only query (defaults to NotImplementedError if unused)."""
        raise NotImplementedError("Text-only query is not implemented for this ObserverBackend")


class StateCorrector(ABC):
    """Lightweight model-specific state corrector attached to a frozen GeneratorAdapter."""

    @abstractmethod
    def set_active(self, active: bool) -> None:
        """Toggle corrector weights on/off without mutating frozen base weights."""

    @abstractmethod
    def trainable_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Canonical name -> Parameter mapping for optimizer and checkpointing."""

    @abstractmethod
    def state_dict_trainable(self) -> dict[str, torch.Tensor]:
        """Minimal serializable checkpoint containing only corrector parameters."""

    @abstractmethod
    def load_state_dict_trainable(self, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
        """Restore corrector parameters into an already-attached generator."""
