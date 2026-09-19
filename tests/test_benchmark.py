from __future__ import annotations

from app.services.benchmark import aggregate, benchmark_scenarios
from app.services.orchestrator import compute_delta


def test_aggregate_empty():
    assert aggregate([]) == {}


def _sample_result():
    gma = {"model": "llama-3-8b", "adapter": "lora-code", "tokens_in": 30, "tokens_out": 40,
           "latency_ms": 200.0, "energy_kwh": 0.01, "carbon_g": 5.0, "cost_usd": 5e-5,
           "quality": 0.9, "depth_used": 0.75}
    base = {"model": "llama-3-8b", "tokens_in": 60, "tokens_out": 50,
            "latency_ms": 350.0, "energy_kwh": 0.02, "carbon_g": 12.0, "cost_usd": 1e-4,
            "quality": 0.88, "depth_used": 1.0}
    return {"gma": gma, "baseline": base}


def test_aggregate_single():
    results = [{"gma": _sample_result()["gma"], "baseline": _sample_result()["baseline"]}]
    out = aggregate(results)
    assert out["scenarios_run"] == 1
    assert "reductions_pct" in out
    assert out["reductions_pct"]["carbon"] < 0  # carbon reduced
    assert out["reductions_pct"]["tokens"] < 0  # fewer tokens
    assert "carbon_saved_g" in out
    assert out["model_usage"]["models"]["llama-3-8b"] == 1


def test_compute_delta():
    opt = {
        "structured_metrics": {
            "tokens_in": 30, "latency_ms": 200.0, "energy_kwh": 0.01,
            "carbon_kg": 0.005, "cost_usd": 5e-5, "quality_score": 0.9,
            "depth_used": 0.75, "carbon_g": 5.0,
        }
    }
    base = {
        "tokens_in": 60, "latency_ms": 350.0, "energy_kwh": 0.02,
        "carbon_kg": 0.012, "cost_usd": 1e-4, "quality_score": 0.88,
        "depth_used": 1.0, "carbon_g": 12.0,
    }
    d = compute_delta(opt, base)
    assert d["token_reduction_pct"] == 50.0
    assert d["carbon_saved_g"] == 7.0
    assert d["energy_saved_kwh"] == 0.01
    assert d["carbon_change_pct"] < 0
    assert d["energy_change_pct"] < 0


def test_benchmark_scenarios(db_session):
    out = benchmark_scenarios(
        scenarios=[
            {"prompt": "What is the capital of France?"},
            {"prompt": "Write a short Python loop."},
        ],
        name="unit-test-benchmark",
        save_records=False,
        db=db_session,
    )
    assert out["scenarios"] == 2
    assert len(out["results"]) == 2
    assert "reductions_pct" in out["summary"]
    for r in out["results"]:
        assert r["gma"]["tokens_in"] > 0
        assert r["gma"]["tokens_in"] < r["baseline"]["tokens_in"] or r["gma"]["model"] != r["baseline"]["model"]
        assert "delta" in r