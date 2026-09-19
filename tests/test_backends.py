from __future__ import annotations

from app.backends.simulator import simulator_backend
from app.backends.base import InferenceResult


def test_simulator_deterministic():
    r1 = simulator_backend.run("What is green AI?", "llama-3-8b", max_tokens=32, early_exit=True)
    r2 = simulator_backend.run("What is green AI?", "llama-3-8b", max_tokens=32, early_exit=True)
    assert isinstance(r1, InferenceResult)
    assert r1.text == r2.text
    assert r1.tokens_in == r2.tokens_in
    assert r1.tokens_out == r2.tokens_out
    assert r1.latency_ms > 0


def test_simulator_early_exit_speedup():
    full = simulator_backend.run("Explain recursion in detail with examples", "llama-3-8b",
                                 max_tokens=64, early_exit=True, depth_used=1.0)
    half = simulator_backend.run("Explain recursion in detail with examples", "llama-3-8b",
                                 max_tokens=64, early_exit=True, depth_used=0.5)
    assert half.latency_ms < full.latency_ms
    assert half.depth_used == 0.5


def test_simulator_varied_domains():
    finance = simulator_backend.run("Calculate portfolio returns with interest", "llama-3-8b", max_tokens=32)
    medical = simulator_backend.run("Diagnose symptoms for treatment", "llama-3-8b", max_tokens=32)
    assert len(finance.text) > 0
    assert len(medical.text) > 0
    assert finance.measured["domain"] == "finance"
    assert medical.measured["domain"] == "medical"


def test_simulator_probe():
    assert simulator_backend.available()


def test_inference_result_to_dict():
    r = InferenceResult(text="hi", tokens_in=1, tokens_out=2, latency_ms=5.0, depth_used=0.5)
    d = r.to_dict()
    assert d["text"] == "hi"
    assert d["depth_used"] == 0.5