"""Adapter for hosted GLM-5.3-flash vision calls via Metto (/v1/chat/completions)."""

import os
from pathlib import Path

from qwen_latent_cot.count_spatial import vlm as vlm_mod
from qwen_latent_cot.count_spatial.vlm import VlmClient

DEFAULT_METTO_BASE = "https://metto-api.com"
DEFAULT_BASE_URL = "https://metto-api.com/v1/chat/completions"
DEFAULT_MODEL = "glm-5.3-flash"
KEY_ENV = "METTO_GLM_KEY"
LEGACY_KEY_ENV = "METTO_KEY"


def resolve_glm_endpoint(cfg_url=None):
    """Resolve full Metto /v1/chat/completions URL, ignoring local 127.0.0.1 defaults."""
    raw = str(cfg_url or "").strip()
    if not raw or "127.0.0.1" in raw or "localhost" in raw:
        raw = os.environ.get("METTO_BASE_URL", DEFAULT_METTO_BASE).strip()
    raw = raw.rstrip("/")
    if not raw.endswith("/v1/chat/completions"):
        raw = raw + "/v1/chat/completions"
    return raw


def resolve_glm_key(cfg_key_env=None):
    """Read Metto GLM API key from configured env var, METTO_GLM_KEY, or METTO_KEY."""
    candidates = []
    if cfg_key_env:
        candidates.append(str(cfg_key_env))
    candidates.extend([KEY_ENV, LEGACY_KEY_ENV])
    for name in candidates:
        val = os.environ.get(name, "").strip()
        if val and not val.startswith("REPLACE_WITH_"):
            return val
    return ""


class GlmClient(VlmClient):
    """A VlmClient backed by Metto's ``glm-5.3-flash`` chat-completions endpoint."""

    def __init__(self, api_key, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL, **kwargs):
        if not api_key and kwargs.get("transport") is None:
            raise ValueError("The hosted GLM model needs a key in " + KEY_ENV)
        super().__init__(base_url=base_url, model=model, api_key=api_key, **kwargs)


def make(cfg, root, stage, transport=None, client=VlmClient):
    """Construct the configured vision client for one pipeline stage."""
    traces = vlm_mod.open_traces(Path(root) / "logs" / "vlm", stage)
    common = dict(
        workers=cfg.get("workers", 16),
        retries=cfg.get("retries", 3),
        timeout=cfg.get("timeout", 180),
        max_tokens=cfg.get("max_tokens", 4096),
        traces=traces,
        transport=transport,
    )
    if client is not VlmClient:
        return client(
            cfg.get("vlm_url", vlm_mod.DEFAULT_BASE_URL),
            cfg.get("vlm_model", vlm_mod.DEFAULT_MODEL),
            **common,
        )
    if cfg.get("vlm_kind", "local") == "glm":
        raw_model = str(cfg.get("vlm_model") or "").strip()
        model = DEFAULT_MODEL if (not raw_model or "qwen" in raw_model.lower()) else raw_model
        return GlmClient(
            resolve_glm_key(cfg.get("vlm_key_env")),
            resolve_glm_endpoint(cfg.get("vlm_url")),
            model,
            **common,
        )
    return VlmClient(
        cfg.get("vlm_url", vlm_mod.DEFAULT_BASE_URL),
        cfg.get("vlm_model", vlm_mod.DEFAULT_MODEL),
        api_key=os.environ.get(cfg.get("vlm_key_env") or "", ""),
        **common,
    )
