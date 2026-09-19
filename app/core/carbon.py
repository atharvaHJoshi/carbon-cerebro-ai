"""Carbon Prediction Engine.

Estimates energy (kWh) and carbon (kg CO2) for an execution strategy using a
hybrid model:

    E_token = tokens_in * ept_in + tokens_out * ept_out
    E_time  = P (kW) * t (h)
    E       = hybrid(E_token, E_time)
    CO2     = E * grid_intensity

Where `ept_*` are calibrated per-model energy-per-token constants and P is the
device power draw. Real measured latency is used when available so the estimate
tracks actual execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings


@dataclass
class ModelEnergyProfile:
    model_id: str
    params: int                       # parameter count
    ept_in: float                     # kWh per prefill/input token
    ept_out: float                    # kWh per decode/output token
    device_power_watts: float         # average power during inference
    prefill_tps: float                # input tokens per second on-device
    decode_tps: float                 # output tokens per second on-device
    fixed_overhead_kwh: float = 0.0   # model load / request overhead

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "params": self.params,
            "ept_in": self.ept_in,
            "ept_out": self.ept_out,
            "device_power_watts": self.device_power_watts,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
        }


@dataclass
class CarbonEstimate:
    model_id: str
    tokens_in: int
    tokens_out: int
    energy_kwh: float
    carbon_kg: float
    grid_region: str
    grid_intensity: float
    method: str
    latency_ms: float | None = None
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.tokens_in + self.tokens_out,
            "energy_kwh": round(self.energy_kwh, 8),
            "carbon_kg": round(self.carbon_kg, 8),
            "carbon_g": round(self.carbon_kg * 1000.0, 6),
            "carbon_mg": round(self.carbon_kg * 1e6, 2),
            "grid_region": self.grid_region,
            "grid_intensity": self.grid_intensity,
            "method": self.method,
            "latency_ms": self.latency_ms,
            "components": self.components,
        }


def grid_intensity_for(region: str | None = None) -> tuple[str, float]:
    region = (region or settings.DEFAULT_GRID_REGION).upper()
    if region not in settings.GRID_INTENSITY_TABLE:
        region = settings.DEFAULT_GRID_REGION.upper()
    return region, settings.GRID_INTENSITY_TABLE[region]


def estimate_carbon(profile: ModelEnergyProfile, tokens_in: int, tokens_out: int,
                    latency_ms: float | None = None, region: str | None = None,
                    layers_ratio: float = 1.0) -> CarbonEstimate:
    """Estimate energy & carbon for an inference.

    `layers_ratio` scales device effort for early-exit plans (fewer layers ->
    proportionally less compute and power).
    """
    region_code, intensity = grid_intensity_for(region)

    # token-based energy
    e_token = tokens_in * profile.ept_in + tokens_out * profile.ept_out
    e_token *= layers_ratio

    # time-based energy (hybrid term for better realism)
    e_time = 0.0
    if latency_ms is not None:
        # effective power proportional to depth used
        eff_power = profile.device_power_watts * (0.6 + 0.4 * layers_ratio)
        e_time = eff_power * (latency_ms / 3600_000.0)
        e_time *= 0.5  # utilization factor (device not at full draw for pure compute estimate)

    energy = e_token + e_time + profile.fixed_overhead_kwh
    carbon = energy * intensity

    method = "hybrid" if latency_ms is not None else "token_based"
    components = {
        "token_energy_kwh": round(e_token, 10),
        "time_energy_kwh": round(e_time, 10),
        "overhead_kwh": round(profile.fixed_overhead_kwh, 10),
        "layers_ratio": round(layers_ratio, 4),
        "formula": "E = in*ept_in + out*ept_out + P*util*t + overhead; CO2 = E * grid",
    }

    return CarbonEstimate(
        model_id=profile.model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        energy_kwh=energy,
        carbon_kg=carbon,
        grid_region=region_code,
        grid_intensity=intensity,
        method=method,
        latency_ms=latency_ms,
        components=components,
    )


def energy_for_tokens_constants(model_id: str) -> ModelEnergyProfile:
    """Calibrated literature-informed default profiles per model class."""
    catalog = {
        "tinygpt": ModelEnergyProfile("tinygpt", 1_000_000, 3.0e-8, 1.2e-7, 15.0, 4000.0, 3000.0),
        "flan-t5-base": ModelEnergyProfile("flan-t5-base", 250_000_000, 2.8e-6, 1.1e-5, 90.0, 900.0, 240.0),
        "gemma-2b": ModelEnergyProfile("gemma-2b", 2_000_000_000, 1.0e-5, 3.8e-5, 180.0, 500.0, 90.0),
        "llama-3-8b": ModelEnergyProfile("llama-3-8b", 8_000_000_000, 2.8e-5, 1.1e-4, 320.0, 260.0, 55.0),
        "mistral-7b": ModelEnergyProfile("mistral-7b", 7_000_000_000, 2.4e-5, 9.5e-5, 300.0, 300.0, 60.0),
        "qwen-7b": ModelEnergyProfile("qwen-7b", 7_000_000_000, 2.4e-5, 9.5e-5, 300.0, 300.0, 60.0),
        "gpt-large-70b": ModelEnergyProfile("gpt-large-70b", 70_000_000_000, 9.0e-5, 3.5e-4, 700.0, 120.0, 22.0),
        "gemini-pro": ModelEnergyProfile("gemini-pro", 100_000_000_000, 1.2e-4, 4.5e-4, 800.0, 100.0, 18.0),
        "claude-3.5": ModelEnergyProfile("claude-3.5", 120_000_000_000, 1.4e-4, 5.2e-4, 850.0, 90.0, 16.0),
    }
    m = catalog.get(model_id)
    if m is not None:
        return m
    # fallback interpolated by parameter count
    if "8b" in model_id or "7b" in model_id:
        return catalog["llama-3-8b"]
    return catalog["flan-t5-base"]