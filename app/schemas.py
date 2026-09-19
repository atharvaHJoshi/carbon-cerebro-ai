from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt / query")
    history: Optional[list[dict]] = Field(None, description="Conversation history [{'role','content'}, ...]")
    model: Optional[str] = Field(None, description="Force a specific model (else router decides)")
    adapter: Optional[str] = Field(None, description="Force a specific LoRA adapter id (else router decides)")
    max_tokens: int = Field(128, ge=8, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    use_pruning: bool = Field(True)
    use_context_compression: bool = Field(True)
    use_early_exit: bool = Field(True)
    use_lora: bool = Field(True)
    strategy: Literal["auto", "full", "minimal"] = Field("auto")
    region: Optional[str] = Field(None, description="Grid region code e.g. IND, USA")
    weights: Optional[dict[str, float]] = Field(None, description="Objective weights quality/carbon/energy/cost/latency/resource")
    constraints: Optional[dict[str, Any]] = Field(None, description="min_quality / max_carbon_g / max_latency_ms / max_cost_usd / exclude_models")
    save_record: bool = Field(True, description="Persist request + analytics record")
    reference: Optional[str] = Field(None, description="Optional reference answer for quality evaluation")


class BaselineRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: Optional[list[dict]] = Field(None)
    model: Optional[str] = Field(None, description="Default baseline model (else llama-3-8b)")
    max_tokens: int = Field(128, ge=8, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    region: Optional[str] = Field(None)
    save_record: bool = Field(True)
    reference: Optional[str] = Field(None)


class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: Optional[list[dict]] = Field(None)


class BenchmarkRequest(BaseModel):
    scenarios: list[dict] = Field(default_factory=list, description="List of {'prompt','history','reference','max_tokens'}.")
    default_model: str = Field("llama-3-8b")
    baseline_model: Optional[str] = Field(None)
    name: str = Field("automated-benchmark")
    region: Optional[str] = Field(None)
    save_records: bool = Field(True)


class EstimateRequestLegacy(BaseModel):
    """Backwards-compatible single carbon estimate endpoint."""
    prompt: str = Field(..., min_length=1)
    model_name: str = Field("llama-3-8b")
    max_tokens: int = Field(128, ge=1)
    region: Optional[str] = Field(None)
    simulate: bool = Field(True, description="ignored; estimation is always modeled/simulated here")


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: Optional[list[dict]] = Field(None)
    weights: Optional[dict[str, float]] = Field(None)
    constraints: Optional[dict[str, Any]] = Field(None)

    class Config:
        extra = "allow"