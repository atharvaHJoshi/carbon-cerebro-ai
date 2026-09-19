"""Inference backend abstraction (interface contract)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class InferenceResult:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    depth_used: float            # 0..1 ratio of layers executed
    measured: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": round(self.latency_ms, 1),
            "depth_used": round(self.depth_used, 4),
            "measured": self.measured,
            "metrics": self.metrics,
        }


class InferenceBackend(ABC):
    """All backends must expose this. Model is selected by `model_id`."""

    @abstractmethod
    def run(self, prompt: str, model_id: str, *, max_tokens: int = 128,
            early_exit: bool = True, temperature: float = 0.7) -> InferenceResult:
        ...

    def available(self) -> bool:
        try:
            self._probe()
            return True
        except Exception:
            return False

    def _probe(self) -> None:
        raise NotImplementedError