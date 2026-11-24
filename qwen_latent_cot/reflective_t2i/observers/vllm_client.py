"""OpenAI-compatible HTTP VLM Observer client (works with vLLM/SGLang Qwen3.5, Qwen3-VL, InternVL3.5)."""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Any

from PIL import Image

from ..core.contracts import ObserverBackend


def _pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


class OpenAIVLMObserver(ObserverBackend):
    """Calls any OpenAI-compatible chat completions endpoint (vLLM / SGLang / API)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("VLM_API_KEY", "EMPTY")
        self.timeout_seconds = float(timeout_seconds)
        self.extra_body = dict(extra_body or {})

    def _post(self, payload: dict[str, Any]) -> str:
        url = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
        body = json.dumps({**self.extra_body, **payload}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()

    def query_image(
        self,
        image: Image.Image,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return self._post(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": int(max_new_tokens),
                "temperature": float(temperature),
            }
        )

    def query_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._post(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": int(max_new_tokens),
                "temperature": float(temperature),
            }
        )
