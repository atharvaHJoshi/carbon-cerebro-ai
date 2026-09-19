from __future__ import annotations

import pytest

from app.core.carbon import (
    ModelEnergyProfile,
    energy_for_tokens_constants,
    estimate_carbon,
    grid_intensity_for,
)
from app.config import settings


def test_grid_intensity_default():
    region, intensity = grid_intensity_for()
    assert region == settings.DEFAULT_GRID_REGION
    assert intensity == settings.GRID_INTENSITY_TABLE[settings.DEFAULT_GRID_REGION]


def test_grid_intensity_france():
    region, intensity = grid_intensity_for("FRA")
    assert region == "FRA"
    assert intensity == pytest.approx(0.052)


def test_grid_intensity_unknown_falls_back():
    region, intensity = grid_intensity_for("ZZZ")
    assert region == settings.DEFAULT_GRID_REGION


def test_profiles_exist():
    for mid in ["tinygpt", "llama-3-8b", "mistral-7b", "qwen-7b", "gpt-large-70b"]:
        p = energy_for_tokens_constants(mid)
        assert p.ept_in > 0
        assert p.ept_out > 0
        assert p.prefill_tps > 0
        assert p.decode_tps > 0


def test_unknown_model_falls_back():
    p = energy_for_tokens_constants("does-not-exist")
    assert p.ept_in > 0


def test_estimate_carbon_monotonic():
    p = energy_for_tokens_constants("llama-3-8b")
    small = estimate_carbon(p, 50, 20, latency_ms=100.0, region="IND")
    large = estimate_carbon(p, 500, 200, latency_ms=1000.0, region="IND")
    assert large.energy_kwh > small.energy_kwh
    assert large.carbon_kg > small.carbon_kg


def test_estimate_carbon_zero_tokens():
    p = energy_for_tokens_constants("llama-3-8b")
    est = estimate_carbon(p, 0, 0, latency_ms=50.0, region="IND")
    assert est.energy_kwh >= 0.0
    assert est.carbon_kg >= 0.0


def test_estimate_carbon_france_lower():
    p = energy_for_tokens_constants("llama-3-8b")
    ind = estimate_carbon(p, 100, 50, latency_ms=200.0, region="IND")
    fra = estimate_carbon(p, 100, 50, latency_ms=200.0, region="FRA")
    # France grid is much cleaner
    assert fra.carbon_kg < ind.carbon_kg


def test_layers_ratio_scales_carbon():
    p = energy_for_tokens_constants("llama-3-8b")
    full = estimate_carbon(p, 100, 50, latency_ms=200.0, region="IND", layers_ratio=1.0)
    partial = estimate_carbon(p, 100, 50, latency_ms=200.0, region="IND", layers_ratio=0.5)
    assert partial.energy_kwh < full.energy_kwh


def test_to_dict_fields():
    p = energy_for_tokens_constants("llama-3-8b")
    est = estimate_carbon(p, 100, 50, latency_ms=200.0, region="IND")
    d = est.to_dict()
    assert d["energy_kwh"] >= 0
    assert d["carbon_kg"] >= 0
    assert d["total_tokens"] == 150
    assert "components" in d


def test_file_profile_roundtrip():
    p = energy_for_tokens_constants("tinygpt")
    d = p.to_dict()
    assert d["params"] == 1_000_000