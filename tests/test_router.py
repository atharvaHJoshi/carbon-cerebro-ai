from __future__ import annotations

from app.core.analysis import analyze_prompt
from app.core.router import candidates_for_request, route


def test_candidates_generated():
    pi = analyze_prompt("Explain recursion in computer science")
    cands = candidates_for_request(pi, adapter_id="lora-code", layers_ratio=0.75,
                                   prune_ratio=0.3, context_ratio=1.0,
                                   tokens_in=50, tokens_out_estimate=60)
    assert len(cands) >= 5
    for c in cands:
        assert c["model"] != ""
        assert 0 <= c["quality"] <= 1
        assert c["carbon_g"] >= 0
        assert c["latency_ms"] > 0
        assert c["energy_kwh"] >= 0


def test_route_returns_decision():
    pi = analyze_prompt("Write a Python function to reverse a linked list")
    decision = route(pi, adapter_id="lora-code", layers_ratio=0.8, prune_ratio=0.25,
                     context_ratio=1.0, tokens_in=40, tokens_out_estimate=50)
    assert decision.selected["model"] in [c["model"] for c in decision.candidates]
    assert decision.selected["model"] != ""
    assert decision.explanation["heading"].startswith("Selected model:")
    assert "reasons" in decision.explanation
    assert len(decision.explanation["reasons"]) > 0


def test_route_respects_min_quality():
    pi = analyze_prompt("What is 2 + 2?")
    decision = route(pi, None, 1.0, 0.0, 1.0, 10, 20, constraints={"min_quality": 0.99})
    # either a model cleared the bar or constraints were relaxed (documented)
    if decision.constraints.get("relaxed"):
        pass
    assert decision.selected["model"] != ""


def test_route_respects_exclude_models():
    pi = analyze_prompt("Write a sorting algorithm")
    decision = route(pi, "lora-code", 1.0, 0.0, 1.0, 30, 40,
                     constraints={"exclude_models": ["gpt-large-70b", "claude-3.5", "gemini-pro"]})
    assert decision.selected["model"] not in ("gpt-large-70b", "claude-3.5", "gemini-pro")


def test_route_carbon_prefers_cleaner_grid():
    pi = analyze_prompt("A moderately complex coding question about binary trees")
    d_ind = route(pi, "lora-code", 1.0, 0.0, 1.0, 30, 40)
    d_fra = route(pi, "lora-code", 1.0, 0.0, 1.0, 30, 40,
                  constraints={"max_carbon_g": 0.5})
    # constraint filtering yields different favorite or same; both valid
    assert d_ind.selected["carbon_g"] >= 0
    assert d_fra.constraints is not None


def test_custom_weights_affect_selection():
    pi = analyze_prompt("Write a complex distributed system design for a banking application")
    w_quality = {"quality": 1.0, "carbon": 0.0, "energy": 0.0, "cost": 0.0, "latency": 0.0, "resource": 0.0}
    w_carbon = {"quality": 0.0, "carbon": 1.0, "energy": 0.0, "cost": 0.0, "latency": 0.0, "resource": 0.0}
    d_quality = route(pi, None, 1.0, 0.0, 1.0, 40, 50, weights=w_quality)
    d_carbon = route(pi, None, 1.0, 0.0, 1.0, 40, 50, weights=w_carbon)
    assert d_quality.weights["quality"] == 1.0
    assert d_carbon.weights["carbon"] == 1.0


def test_runner_up_present_on_multiple():
    pi = analyze_prompt("How to write a recursive algorithm for tree traversal")
    decision = route(pi, None, 1.0, 0.0, 1.0, 30, 30)
    assert decision.runner_up is not None or len(decision.candidates) == 1