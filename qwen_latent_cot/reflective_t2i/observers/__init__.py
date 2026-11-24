"""Decoupled external VLM Observer backends (vLLM OpenAI HTTP & local HF Transformers)."""

from .hf_vlm import LocalHFObserver
from .vllm_client import OpenAIVLMObserver

__all__ = ["LocalHFObserver", "OpenAIVLMObserver"]
