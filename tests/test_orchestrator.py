from __future__ import annotations

from app.schemas import BaselineRequest, GenerateRequest
from app.services.orchestrator import (
    build_execution_prompt,
    compute_delta,
    run_generate,
)


def test_run_generate_end_to_end(db_session, sample_prompt):
    req = GenerateRequest(prompt=sample_prompt, max_tokens=32, save_record=False)
    out = run_generate(req, db=db_session)
    assert out["request_id"]
    assert out["prompt_intelligence"]["intent"] != ""
    assert out["optimizations"]["token_pruning"] is not None
    assert out["decision"]["selected"]["model"]
    assert out["result"]["response"]
    assert out["result"]["structured_metrics"]["carbon_g"] >= 0
    assert out["comparison"]["baseline"]["tokens_in"] > out["result"]["structured_metrics"]["tokens_in"]


def test_run_generate_saves_record(db_session, simple_prompt):
    req = GenerateRequest(prompt=simple_prompt, max_tokens=16, save_record=True)
    out = run_generate(req, db=db_session)
    assert out["record_id"] is not None


def test_run_generate_with_history(db_session, sample_prompt, conversation_history):
    req = GenerateRequest(prompt=sample_prompt, history=conversation_history, max_tokens=32, save_record=False)
    out = run_generate(req, db=db_session)
    assert out["optimizations"]["context_compression"] is not None
    comp = out["optimizations"]["context_compression"]
    assert comp["retention"] <= 1.0


def test_run_generate_disabled_optimizations(db_session, sample_prompt):
    req = GenerateRequest(
        prompt=sample_prompt, max_tokens=32, save_record=False,
        use_pruning=False, use_context_compression=False, use_early_exit=False, use_lora=False,
    )
    out = run_generate(req, db=db_session)
    assert out["optimizations"]["token_pruning"] is None


def test_run_generate_with_reference(db_session, sample_prompt):
    ref = (f"To solve this problem one would implement a function using the Fibonacci recurrence "
           f"and dynamic programming with memoization to avoid redundant computation.")
    req = GenerateRequest(prompt=sample_prompt, max_tokens=32, reference=ref, save_record=False)
    out = run_generate(req, db=db_session)
    assert out["result"]["evaluation"]["quality_score"] >= 0


def test_build_execution_prompt():
    from app.core.context import Message
    out = build_execution_prompt("hi there", [Message("user", "hello"), Message("assistant", "hey")])
    assert "User: hi there" in out
    assert "hello" in out


def test_compute_delta_zeros():
    opt = {"structured_metrics": {"tokens_in": 0, "latency_ms": 0.0, "energy_kwh": 0.0,
                                  "carbon_kg": 0.0, "cost_usd": 0.0, "quality_score": 0.0,
                                  "depth_used": 0.0, "carbon_g": 0.0}}
    base = {"tokens_in": 0, "latency_ms": 0.0, "energy_kwh": 0.0, "carbon_kg": 0.0,
            "cost_usd": 0.0, "quality_score": 0.0, "depth_used": 0.0, "carbon_g": 0.0}
    d = compute_delta(opt, base)
    # no zero-division errors
    assert d["token_reduction_pct"] == 0.0
    assert d["carbon_saved_g"] == 0.0