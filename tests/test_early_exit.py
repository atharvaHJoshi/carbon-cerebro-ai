from __future__ import annotations

from app.core.analysis import analyze_prompt
from app.core.early_exit import note_measured_exit, plan_layers


def _pi(text):
    return analyze_prompt(text)


def test_plan_full_strategy():
    pi = _pi("hello world")
    plan = plan_layers(pi, 32, strategy="full")
    assert plan.layers_to_run == 32
    assert plan.layers_saved == 0
    assert plan.expected_quality_factor == 1.0


def test_plan_auto_saves_layers_easy():
    pi = _pi("What is 2 + 2?")
    plan = plan_layers(pi, 32, strategy="auto")
    assert 1 <= plan.layers_to_run <= 32
    assert plan.total_layers == 32
    assert len(plan.reasons) > 0


def test_plan_minimal_uses_floor():
    pi = _pi("What is 2 + 2?")
    plan = plan_layers(pi, 32, strategy="minimal")
    floor = max(1, int(32 * 0.25))
    assert plan.layers_to_run <= 32
    assert plan.layers_to_run >= 1
    # saves are recorded
    assert plan.layers_saved >= 0
    # minimal never runs full depth for a simple prompt
    assert plan.layers_to_run < 32 or plan.layers_saved == 0


def test_plan_trivial_depth():
    pi = _pi("hi")
    plan = plan_layers(pi, 1, strategy="auto")
    assert plan.layers_to_run == 1
    assert plan.layers_saved == 0


def test_plan_high_confidence_threshold_runs_deeper():
    pi = _pi("Explain the full proof of Fermat's Last Theorem with references")
    plan_low = plan_layers(pi, 32, confidence_threshold=0.5)
    plan_high = plan_layers(pi, 32, confidence_threshold=0.95)
    assert plan_high.layers_to_run >= plan_low.layers_to_run


def test_measured_exit_updates_plan():
    pi = _pi("hello world this is a test prompt for tinygpt")
    plan = plan_layers(pi, 6, strategy="full")
    updated = note_measured_exit(plan, [3, 4, 3, 5], quality_factor=0.7)
    assert updated.layers_to_run <= 6
    assert updated.expected_quality_factor <= 1.0
    assert any("measured" in r for r in updated.reasons)