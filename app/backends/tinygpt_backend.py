"""tinygpt adapter - real from-scratch transformer with measured early exit."""

from __future__ import annotations

from typing import Any

from ..core.carbon import energy_for_tokens_constants
from ..tokenizer import count_tokens
from .base import InferenceBackend, InferenceResult
from .tinygpt import get_tinygpt


class TinyGPTBackend(InferenceBackend):
    def _probe(self) -> None:
        m = get_tinygpt()
        m.generate("the", max_new=2, early_exit=False)

    def run(self, prompt: str, model_id: str = "tinygpt", *, max_tokens: int = 32,
            early_exit: bool = True, temperature: float = 0.8, confidence_threshold: float | None = None,
            **kw: Any) -> InferenceResult:
        import time

        m = get_tinygpt()
        profile = energy_for_tokens_constants("tinygpt")
        tokens_in = count_tokens(prompt)

        start = time.time()
        res = m.generate(prompt, max_new=min(max_tokens, m.cfg["block"] // 2),
                         temperature=temperature, early_exit=early_exit,
                         confidence_threshold=confidence_threshold)
        wall_ms = (time.time() - start) * 1000.0

        mean_exit = res["mean_exit_layer"] / max(1, res["layers_total"])
        base_latency = tokens_in / profile.prefill_tps * 1000.0
        depth_latency = wall_ms - base_latency
        latency_ms = max(wall_ms, base_latency * 0.5 + wall_ms * 0.5)

        return InferenceResult(
            text=res["text"],
            tokens_in=tokens_in,
            tokens_out=res["tokens_out"],
            latency_ms=latency_ms,
            depth_used=mean_exit,
            measured={
                "early_exit": early_exit,
                "mean_exit_layer": res["mean_exit_layer"],
                "layers_total": res["layers_total"],
                "mean_confidence": res["mean_confidence"],
                "exit_layers": res["exit_layers"][:40],
                "confidences": res["confidences"][:40],
                "real_transformer": True,
            },
        )


tinygpt_backend = TinyGPTBackend()