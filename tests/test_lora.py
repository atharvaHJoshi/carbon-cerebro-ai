from __future__ import annotations

from app.core.lora import (
    ADAPTER_CATALOG,
    adapter_energy_overhead,
    adapter_quality_boost,
    select_adapter,
)


def test_catalog_shape():
    assert len(ADAPTER_CATALOG) >= 4
    ids = [a["id"] for a in ADAPTER_CATALOG]
    assert "lora-code" in ids
    assert "lora-finance" in ids
    assert "lora-general" in ids


def test_select_code_adapter():
    decision = select_adapter("Write a python function to sort a list of dictionaries by key")
    assert decision.candidates, "candidates list must be populated"
    best = decision.candidates[0]
    assert best["id"] == "lora-code"


def test_select_finance_adapter():
    decision = select_adapter("Compute the compound interest on my stock portfolio investment")
    best = decision.candidates[0]
    assert best["id"] == "lora-finance"


def test_select_medical_adapter():
    decision = select_adapter("Diagnose treatment options for a patient with diabetes symptoms")
    best = decision.candidates[0]
    assert best["id"] == "lora-medical"


def test_pipeline_records_steps():
    decision = select_adapter("How to fix a memory leak in a python daemon")
    assert isinstance(decision.pipeline, list)
    assert len(decision.pipeline) >= 1
    assert decision.pipeline[0]["step"] == "adapter-selection"


def test_fallback_base_for_general():
    decision = select_adapter("What is your name and today's date? Tell me a fun fact.")
    assert decision.selected is None or decision.fallback_to_base is False
    assert decision.reasons is not None


def test_adapter_helpers():
    assert adapter_energy_overhead(None) == 0.0
    assert adapter_energy_overhead("lora-code") > 0.0
    assert adapter_quality_boost(None, 0.5) == 0.5
    assert adapter_quality_boost("lora-code", 0.5) > 0.5