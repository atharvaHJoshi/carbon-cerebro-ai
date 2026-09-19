"""Live remote-model backend (OpenAI / vLLM-compatible chat completions).

Pluggable: if `GMA_OPENAI_BASE_URL` and `GMA_OPENAI_API_KEY` are set, requests
route to a real hosted model (e.g. a local vLLM server) and carbon is estimated
from real latency + token counts. Without config this backend is unavailable and
the platform gracefully falls back to the simulator.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import settings
from ..tokenizer import count_tokens
from .base import InferenceBackend, InferenceResult

BASE_URL = os.getenv("GMA_OPENAI_BASE_URL", "")
API_KEY = os.getenv("GMA_OPENAI_API_KEY", "")
MODEL = os.getenv("GMA_REMOTE_MODEL", "llama-3-8b")


class RemoteBackend(InferenceBackend):
    def available(self) -> bool:
        return bool(BASE_URL and API_KEY)

    def _probe(self) -> None:
        if not self.available():
            raise RuntimeError("remote backend not configured")

    def run(self, prompt: str, model_id: str, *, max_tokens: int = 128,
            early_exit: bool = True, temperature: float = 0.7, **kw: Any) -> InferenceResult:
        if not self.available():
            raise RuntimeError("remote backend not configured; set GMA_OPENAI_BASE_URL/GMA_OPENAI_API_KEY")
        import requests

        start = time.time()
        resp = requests.post(
            f"{BASE_URL.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=settings.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.time() - start) * 1000.0
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", count_tokens(prompt))
        tokens_out = usage.get("completion_tokens", count_tokens(text))
        return InferenceResult(
            text=text, tokens_in=tokens_in, tokens_out=tokens_out,
            latency_ms=latency_ms, depth_used=1.0,
            measured={"remote": True, "model": MODEL},
        )


remote_backend = RemoteBackend()