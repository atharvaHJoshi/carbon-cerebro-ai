from __future__ import annotations

from app.database import RequestRecord
from app.tokenizer import count_tokens, token_stats, tokenize


def test_count_tokens():
    assert count_tokens("Hello world, this is a test.") > 0
    assert count_tokens(None) == 0
    assert count_tokens("") == 0


def test_token_stats():
    s = token_stats("token counting should work here nicely")
    assert s["tokens"] > 0
    assert s["words"] > 0
    assert s["chars"] > 0
    assert s["method"] in ("tiktoken", "fallback")


def test_tokenize_returns_list():
    toks = tokenize("split me into tokens")
    assert isinstance(toks, list)
    assert len(toks) > 0


def test_request_record_roundtrip():
    d = {
        "kind": "gma",
        "session_id": "abc123",
        "prompt": "hello",
        "model_name": "llama-3-8b",
        "adapter_name": "lora-code",
        "layers_run": 20,
        "layers_total": 32,
        "prune_ratio": 0.3,
        "context_ratio": 1.0,
        "tokens_in": 10,
        "tokens_out": 20,
        "tokens_original": 15,
        "latency_ms": 123.4,
        "energy_kwh": 0.001,
        "carbon_kg": 0.0005,
        "cost_usd": 0.0001,
        "quality_score": 0.9,
        "grid_region": "IND",
        "grid_intensity": 0.708,
        "response": "some output",
        "explanation": {"why": "because"},
        "metrics": {"extra": {"key": 1}},
    }
    rec = RequestRecord.from_dict(d)
    assert rec.to_dict()["response"] == "some output"
    assert rec.to_dict()["explanation"] == {"why": "because"}
    out = rec.to_dict()
    assert out["model_name"] == "llama-3-8b"
    assert out["metrics"]["extra"] == {"key": 1}


def test_request_record_missing_fields():
    rec = RequestRecord.from_dict({"prompt": "x"})
    out = rec.to_dict()
    assert out["prompt"] == "x"
    assert out["kind"] == "gma"
    assert out["prune_ratio"] == 0.0


def test_create_tables(db_session):
    from app.database import RequestRecord
    db_session.add(RequestRecord(prompt="smoke test"))
    db_session.commit()
    rows = db_session.query(RequestRecord).filter(RequestRecord.prompt == "smoke test").all()
    assert len(rows) == 1