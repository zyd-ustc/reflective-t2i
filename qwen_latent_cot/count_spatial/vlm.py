"""Chat-completions client for vision models (local OpenAI-compatible or hosted GLM-5.3-flash)."""

import base64
import json
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8092/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class VlmError(RuntimeError):
    """A call that did not produce a usable reply."""


def norm(text):
    """Comparisons are made on lower-cased words without punctuation."""
    return " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())


def open_traces(trace_path, stage):
    """An append-only JSONL of this stage's prompts and replies, or None."""
    if trace_path in (None, ""):
        return None
    base = Path(trace_path)
    path = base if base.suffix == ".jsonl" else base / (stage + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return {stage: path.open("a", encoding="utf-8")}


def encode_image(path):
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


class VlmClient:
    """Concurrent chat-completions calls that return parsed JSON objects."""

    def __init__(
        self,
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        api_key="",
        workers=16,
        retries=3,
        timeout=180,
        max_tokens=4096,
        transport=None,
        traces=None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.workers = max(1, int(workers))
        self.retries = max(1, int(retries))
        self.timeout = int(timeout)
        self.default_max_tokens = int(max_tokens)
        self._session = None
        if transport is None:
            try:
                import requests
                from requests.adapters import HTTPAdapter

                sess = requests.Session()
                pool_sz = max(32, self.workers * 2)
                adapter = HTTPAdapter(pool_connections=pool_sz, pool_maxsize=pool_sz)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
                self._session = sess
            except ImportError:
                self._session = None
        self.transport = transport or self._http_transport
        self.traces = traces

    def _http_transport(self, messages, max_tokens):
        payload = dict(
            model=self.model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=0,
            stream=False,
        )
        if "glm" in str(self.model).lower():
            payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "curl/8.7.1",
        }
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        if self._session is not None:
            try:
                resp = self._session.post(self.base_url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code >= 400:
                    raise VlmError("HTTP %d: %s" % (resp.status_code, resp.text[:240]))
                reply = resp.json()
            except VlmError:
                raise
            except Exception as exc:
                raise VlmError("transport error: %s" % exc) from exc
        else:
            body = json.dumps(payload).encode()
            request = urllib.request.Request(self.base_url, data=body, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                reply = json.loads(response.read())
        if isinstance(reply, dict) and reply.get("error"):
            raise VlmError("API error: " + json.dumps(reply["error"], ensure_ascii=False))
        return reply["choices"][0]["message"].get("content") or ""

    @staticmethod
    def _decode(content):
        text = re.sub(r"<think>[\s\S]*?</think>", "", str(content or ""), flags=re.I).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            # Also handle prose prefix/suffix around a single JSON object if present
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                try:
                    value = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    raise VlmError("reply is not JSON: " + text[:200]) from exc
            else:
                raise VlmError("reply is not JSON: " + text[:200]) from exc
        if not isinstance(value, dict):
            raise VlmError("reply is not a JSON object: " + text[:200])
        return value

    def call(self, messages, max_tokens=None, check=None, stage=None, uid=None):
        """One JSON object per call, retried on transport errors and rejected replies."""
        limit_tokens = int(self.default_max_tokens if max_tokens is None else max_tokens)
        error, content = None, None
        for attempt in range(self.retries):
            try:
                content = self.transport(messages, limit_tokens)
                value = self._decode(content)
                if check is not None:
                    check(value)
                self._trace(stage, uid, messages, content, ok=True)
                return value
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                VlmError,
                ValueError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                error = exc
                self._trace(stage, uid, messages, "FAILED %s | REPLY %s" % (exc, content), ok=False)
                if attempt + 1 < self.retries:
                    time.sleep(min(15.0, 1.0 * (2**attempt)) + random.random())
        raise VlmError("no usable reply after %d attempts: %s" % (self.retries, error))

    def _trace(self, stage, uid, messages, content, ok):
        handle = (self.traces or {}).get(stage or "default")
        if handle is None:
            return
        prompt = []
        for message in messages:
            parts = message.get("content")
            if isinstance(parts, list):
                parts = [
                    "<image>" if part.get("type") == "image_url" else part.get("text", "")
                    for part in parts
                ]
            else:
                parts = [str(parts)]
            prompt.append(message["role"] + ": " + "\n".join(parts))
        handle.write(json.dumps(dict(uid=uid, ok=ok, prompt=prompt, reply=content)) + "\n")
        handle.flush()

    def call_many(self, jobs):
        """Run ``(uid, messages, kwargs)`` triples concurrently; failures come back in place."""
        results = {}

        def run(job):
            uid, messages, kwargs = job
            try:
                return uid, self.call(messages, **kwargs), None
            except VlmError as exc:
                return uid, None, str(exc)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for uid, value, error in pool.map(run, jobs):
                results[key := uid] = (value, error)
        return results
