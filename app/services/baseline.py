"""Baseline execution - runs the SAME request without any optimization and
persists a comparable record."""

from __future__ import annotations

import uuid

from ..backends.simulator import simulator_backend
from ..core.analysis import analyze_conversation, analyze_prompt
from ..core.carbon import energy_for_tokens_constants, estimate_carbon
from ..core.evaluation import evaluate_response
from ..database import Session, RequestRecord
from ..schemas import BaselineRequest
from ..tokenizer import count_tokens


def run_baseline(req: BaselineRequest, db: Session | None = None) -> dict:
    model = req.model or "llama-3-8b"
    history = req.history or []
    tokens_original = count_tokens(req.prompt) + sum(count_tokens(m.get("content", "")) for m in history)

    # unoptimized execution: raw prompt + full history, full depth, no adapter
    execution_input = req.prompt
    if history:
        msgs = "\n\n".join(f"{m.get('role','user').capitalize()}: {m.get('content','')}" for m in history)
        execution_input = f"{msgs}\n\nUser: {req.prompt}"

    result = simulator_backend.run(execution_input, model, max_tokens=req.max_tokens,
                                   temperature=req.temperature, early_exit=False, depth_used=1.0)

    profile = energy_for_tokens_constants(model)
    carbon = estimate_carbon(profile, result.tokens_in, result.tokens_out,
                             latency_ms=result.latency_ms, region=req.region, layers_ratio=1.0)
    cost_per_tok = {"tinygpt": 0.0, "flan-t5-base": 1e-7, "gemma-2b": 3e-7, "llama-3-8b": 8e-7,
                    "mistral-7b": 7e-7, "qwen-7b": 8e-7, "gpt-large-70b": 5e-6,
                    "gemini-pro": 3e-6, "claude-3.5": 3e-6}.get(model, 8e-7)
    cost = (result.tokens_in + result.tokens_out) * cost_per_tok

    if req.reference:
        eval_ = evaluate_response(result.text, req.reference, latency_ms=result.latency_ms,
                                  tokens_in=result.tokens_in, tokens_out=result.tokens_out)
        quality = eval_.quality_score
    else:
        cap = {"tinygpt": 0.35, "flan-t5-base": 0.55, "gemma-2b": 0.78, "llama-3-8b": 0.9,
               "mistral-7b": 0.88, "qwen-7b": 0.9, "gpt-large-70b": 0.98, "gemini-pro": 0.97,
               "claude-3.5": 0.98}.get(model, 0.9)
        quality = cap * 0.95
        eval_ = evaluate_response(result.text, None, latency_ms=result.latency_ms,
                                  tokens_in=result.tokens_in, tokens_out=result.tokens_out)

    session_id = uuid.uuid4().hex[:12]
    record_id = None
    if db is not None and req.save_record:
        rec = RequestRecord.from_dict({
            "kind": "baseline",
            "session_id": session_id,
            "prompt": req.prompt,
            "model_name": model,
            "layers_run": 32,
            "layers_total": 32,
            "prune_ratio": 0.0,
            "context_ratio": 1.0,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "tokens_original": tokens_original,
            "latency_ms": result.latency_ms,
            "energy_kwh": carbon.energy_kwh,
            "carbon_kg": carbon.carbon_kg,
            "cost_usd": cost,
            "quality_score": quality,
            "grid_region": carbon.grid_region,
            "grid_intensity": carbon.grid_intensity,
            "response": result.text,
            "explanation": {"model": model, "optimizations": {}, "note": "baseline unoptimized run"},
            "metrics": {"tokens_original": tokens_original},
        })
        db.add(rec)
        db.commit()
        db.refresh(rec)
        record_id = rec.id

    return {
        "request_id": session_id,
        "record_id": record_id,
        "metrics": {
            "model": model,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "total_tokens": result.tokens_in + result.tokens_out,
            "tokens_original": tokens_original,
            "latency_ms": round(result.latency_ms, 1),
            "energy_kwh": round(carbon.energy_kwh, 10),
            "carbon_kg": round(carbon.carbon_kg, 10),
            "carbon_g": round(carbon.carbon_kg * 1000, 6),
            "cost_usd": round(cost, 8),
            "quality_score": round(quality, 4),
            "depth_used": 1.0,
            "prune_ratio": 0.0,
            "context_ratio": 1.0,
            "adapter": None,
            "layers_ratio": 1.0,
        },
        "response": result.text,
        "evaluation": eval_.to_dict(),
    }