"""Green Model Advisor orchestration pipeline (analyze -> optimize -> route ->
execute -> measure -> evaluate -> persist)."""

from __future__ import annotations

import time
import uuid

import numpy as np

from ..backends.base import InferenceBackend
from ..backends.remote import remote_backend
from ..backends.simulator import simulator_backend
from ..backends.tinygpt_backend import tinygpt_backend
from ..config import settings
from ..core.analysis import analyze_conversation, analyze_prompt
from ..core.carbon import energy_for_tokens_constants, estimate_carbon
from ..core.context import compress_context
from ..core.early_exit import plan_layers
from ..core.evaluation import evaluate_response
from ..core.lora import adapter_energy_overhead, select_adapter
from ..core.pruning import prune_prompt
from ..core.router import route
from ..database import Session, RequestRecord
from ..schemas import BaselineRequest, GenerateRequest
from ..tokenizer import count_tokens

BASELINE_MODEL = "llama-3-8b"
_REFERENCE_DEPTH = 32  # layer count used for the layer-plan decision


def _backend_for(model: str, use_early_exit: bool) -> InferenceBackend:
    if model == "tinygpt":
        return tinygpt_backend
    if model in ("gemini-pro", "claude-3.5", "gpt-large-70b") and remote_backend.available():
        return remote_backend
    return simulator_backend


def _output_tokens_estimate(pi, model_id: str, max_tokens: int) -> int:
    base = int(24 + 60 * pi.complexity)
    scale = {"tinygpt": 0.5, "flan-t5-base": 0.8}.get(model_id, 1.0)
    return min(max_tokens, max(8, int(base * scale)))


def run_generate(req: GenerateRequest, db: Session | None = None) -> dict:
    session_id = uuid.uuid4().hex[:12]
    steps_start = time.time()

    # ------------------------------------------------ 1. prompt intelligence
    if req.history:
        pi = analyze_conversation([*req.history, {"role": "user", "content": req.prompt}])
    else:
        pi = analyze_prompt(req.prompt)
    tokens_original = count_tokens(req.prompt) + sum(
        count_tokens(m.get("content", "")) for m in (req.history or []))

    # ------------------------------------------------ 2. dynamic token pruning
    pruned = None
    prune_ratio = 0.0
    optimized_prompt = req.prompt
    if req.use_pruning:
        pruned = prune_prompt(req.prompt)
        optimized_prompt = pruned.pruned_text or req.prompt
        prune_ratio = pruned.compression_ratio

    # ------------------------------------------------ 3. context compression
    context_result = None
    context_ratio = 1.0
    if req.use_context_compression and req.history:
        context_result = compress_context(req.history, query=optimized_prompt)
        context_ratio = context_result.retention

    # ------------------------------------------------ 4. adaptive LoRA
    lora = None
    lora_decision = None
    if req.use_lora:
        lora_decision = select_adapter(optimized_prompt)
        lora = lora_decision.selected if not lora_decision.fallback_to_base else None
    if req.adapter:
        lora = req.adapter

    # ------------------------------------------------ 5. layer-wise plan
    total_layers = _REFERENCE_DEPTH
    layer_plan = plan_layers(pi, total_layers, strategy=req.strategy)
    layers_ratio = layer_plan.layers_to_run / max(1, total_layers)

    # ------------------------------------------------ 6. multi-objective routing
    tokens_in_opt = count_tokens(optimized_prompt)
    if context_result is not None:
        tokens_in_opt += context_result.kept_tokens
    else:
        tokens_in_opt += count_tokens(req.prompt) * (1 - prune_ratio)

    profile_overrides = {"tinygpt": energy_for_tokens_constants("tinygpt")}
    decision = route(
        pi, lora, layers_ratio, prune_ratio, context_ratio,
        tokens_in_opt, _output_tokens_estimate(pi, req.model or "llama-3-8b", req.max_tokens),
        weights=req.weights, constraints=req.constraints,
        profile_overrides=profile_overrides, request_id=session_id,
    )
    selected_model = req.model or decision.selected["model"]
    layers_ratio_model = layers_ratio if req.strategy != "full" or req.use_early_exit else 1.0

    # ------------------------------------------------ 7. execute selected config
    backend = _backend_for(selected_model, req.use_early_exit)
    execution_input = optimized_prompt
    if context_result is not None:
        execution_input = build_execution_prompt(optimized_prompt, context_result.messages)

    run_start = time.time()
    backend_kwargs = dict(max_tokens=req.max_tokens, temperature=req.temperature)
    if selected_model == "tinygpt":
        backend_kwargs["early_exit"] = req.use_early_exit
    result = backend.run(execution_input, selected_model, **backend_kwargs)
    execute_ms = (time.time() - run_start) * 1000.0

    depth_used = result.depth_used if result.depth_used is not None else layers_ratio
    if not req.use_early_exit:
        depth_used = 1.0

    # ------------------------------------------------ 8. measure energy/carbon/cost
    profile = energy_for_tokens_constants(selected_model)
    carbon = estimate_carbon(profile, result.tokens_in, result.tokens_out,
                             latency_ms=result.latency_ms or execute_ms,
                             region=req.region, layers_ratio=depth_used)
    cost_per_tok = {"tinygpt": 0.0, "flan-t5-base": 1e-7, "gemma-2b": 3e-7,
                    "llama-3-8b": 8e-7, "mistral-7b": 7e-7, "qwen-7b": 8e-7,
                    "gpt-large-70b": 5e-6, "gemini-pro": 3e-6, "claude-3.5": 3e-6}.get(selected_model, 8e-7)
    cost_usd = (result.tokens_in + result.tokens_out) * cost_per_tok
    if lora and selected_model != "tinygpt":
        energy_kwh_adj = carbon.energy_kwh * (1 + adapter_energy_overhead(lora))
        carbon_adj = energy_kwh_adj * carbon.grid_intensity
    else:
        energy_kwh_adj, carbon_adj = carbon.energy_kwh, carbon.carbon_kg

    # ------------------------------------------------ 9. quality evaluation
    reference = req.reference
    if reference is None and req.history:
        reference = None
    if reference is not None:
        eval_ = evaluate_response(result.text, reference, latency_ms=result.latency_ms,
                                  tokens_in=result.tokens_in, tokens_out=result.tokens_out)
        quality = eval_.quality_score
    else:
        # modeled quality from selected plan
        quality = decision.selected["quality"] * layer_plan.expected_quality_factor
        eval_ = evaluate_response(result.text, None, latency_ms=result.latency_ms,
                                  tokens_in=result.tokens_in, tokens_out=result.tokens_out)
        quality = min(quality, 1.0)

    total_ms = (time.time() - steps_start) * 1000.0

    # ------------------------------------------------ 10. build response bundle
    measured = {
        "execution_time_ms": round(total_ms, 1),
        "backend_latency_ms": round(result.latency_ms or execute_ms, 1),
        "backend": "tinygpt" if selected_model == "tinygpt" else ("remote" if selected_model in ("gemini-pro", "claude-3.5", "gpt-large-70b") and remote_backend.available() else "simulator"),
        "real_transformer": selected_model == "tinygpt",
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "tokens_original": tokens_original,
        "tokens_saved": max(0, tokens_original - result.tokens_in),
        "token_reduction_ratio": round(max(0.0, 1 - result.tokens_in / max(1, tokens_original)), 4),
        "latency_ms": round(result.latency_ms or execute_ms, 1),
        "depth_used": round(depth_used, 4),
        "energy_kwh": round(energy_kwh_adj, 10),
        "carbon_kg": round(carbon_adj, 10),
        "carbon_g": round(carbon_adj * 1000, 6),
        "cost_usd": round(cost_usd, 8),
        "grid_region": carbon.grid_region,
        "grid_intensity": carbon.grid_intensity,
        "quality_score": round(min(quality, 1.0), 4),
    }

    record_id = None
    explanation = decision.explanation
    if db is not None and req.save_record:
        rec = RequestRecord.from_dict({
            "kind": "gma",
            "session_id": session_id,
            "prompt": req.prompt,
            "model_name": selected_model,
            "adapter_name": lora,
            "layers_run": int(round(depth_used * total_layers)),
            "layers_total": total_layers,
            "prune_ratio": prune_ratio,
            "context_ratio": context_ratio,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "tokens_original": tokens_original,
            "latency_ms": result.latency_ms or execute_ms,
            "energy_kwh": energy_kwh_adj,
            "carbon_kg": carbon_adj,
            "cost_usd": cost_usd,
            "quality_score": min(quality, 1.0),
            "grid_region": carbon.grid_region,
            "grid_intensity": carbon.grid_intensity,
            "response": result.text,
            "explanation": explanation,
            "metrics": {"measured": measured, "decision": decision.selected,
                        "layer_plan": layer_plan.to_dict()},
        })
        db.add(rec)
        db.commit()
        db.refresh(rec)
        record_id = rec.id

    baseline = baseline_counterfactual(req, tokens_original=tokens_original, region=req.region)

    return {
        "request_id": session_id,
        "record_id": record_id,
        "prompt_intelligence": pi.to_dict(),
        "optimizations": {
            "token_pruning": pruned.to_dict() if pruned else None,
            "context_compression": context_result.to_dict() if context_result else None,
            "adapter_selection": lora_decision.to_dict() if lora_decision else None,
            "layer_optimization": layer_plan.to_dict(),
            "selected_adapter": lora,
            "prune_ratio": prune_ratio,
            "context_ratio": context_ratio,
            "layers_ratio": round(layers_ratio, 4),
        },
        "execution_plan": {
            "model": selected_model,
            "adapter": lora,
            "input": execution_input,
            "layers_to_run": layer_plan.layers_to_run,
            "layers_total": total_layers,
            "estimated_quality": decision.selected["quality"],
        },
        "decision": decision.to_dict(),
        "result": {
            "response": result.text,
            "structured_metrics": measured,
            "evaluation": eval_.to_dict(),
            "backend_details": result.measured,
        },
        "comparison": {
            "baseline": baseline["metrics"],
            "delta": baseline["delta"],
        },
        "explanation": explanation,
    }


def build_execution_prompt(latest: str, messages) -> str:
    parts = []
    for m in messages:
        role = m.role
        parts.append(f"{role.capitalize()}: {m.content}")
    parts.append(f"User: {latest}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- baseline


def baseline_counterfactual(req: BaselineRequest | GenerateRequest, *, tokens_original: int | None = None,
                            region: str | None = None) -> dict:
    """Numeric baseline: the same request un-optimized on the default model."""
    model = getattr(req, "model", None) or BASELINE_MODEL
    history = getattr(req, "history", None) or []
    prompt = req.prompt
    pi = analyze_conversation([*history, {"role": "user", "content": prompt}]) if history else analyze_prompt(prompt)

    tokens_in = count_tokens(prompt) + sum(count_tokens(m.get("content", "")) for m in history)
    tokens_out = min(req.max_tokens, max(12, int(40 + 60 * pi.complexity)))
    profile = energy_for_tokens_constants(model)
    prefill_ms = tokens_in / profile.prefill_tps * 1000.0
    decode_ms = tokens_out / profile.decode_tps * 1000.0
    latency_ms = prefill_ms + decode_ms

    carbon = estimate_carbon(profile, tokens_in, tokens_out, latency_ms=latency_ms, region=region, layers_ratio=1.0)
    cost_per_tok = {"tinygpt": 0.0, "flan-t5-base": 1e-7, "gemma-2b": 3e-7,
                    "llama-3-8b": 8e-7, "mistral-7b": 7e-7, "qwen-7b": 8e-7,
                    "gpt-large-70b": 5e-6, "gemini-pro": 3e-6, "claude-3.5": 3e-6}.get(model, 8e-7)
    cost = (tokens_in + tokens_out) * cost_per_tok

    cap = {"tinygpt": 0.35, "flan-t5-base": 0.55, "gemma-2b": 0.78, "llama-3-8b": 0.9,
           "mistral-7b": 0.88, "qwen-7b": 0.9, "gpt-large-70b": 0.98, "gemini-pro": 0.97,
           "claude-3.5": 0.98}.get(model, 0.9)
    quality = cap * 0.95

    metrics = {
        "model": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_tokens": tokens_in + tokens_out,
        "latency_ms": round(latency_ms, 1),
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
        "grid_region": carbon.grid_region,
    }
    return {"metrics": metrics, "delta": {}}


def compute_delta(optimized: dict, baseline: dict) -> dict:
    m = optimized.get("structured_metrics", optimized)
    b = baseline.get("metrics", baseline)
    def pct(new, old):
        if not old:
            return 0.0
        return round((new - old) / old * 100.0, 2)

    return {
        "token_reduction_pct": round(pct(m["tokens_in"], b["tokens_in"]) * -1, 2),
        "latency_change_pct": pct(m["latency_ms"], b["latency_ms"]),
        "energy_change_pct": pct(m["energy_kwh"], b["energy_kwh"]),
        "carbon_change_pct": pct(m["carbon_kg"], b["carbon_kg"]),
        "cost_change_pct": pct(m["cost_usd"], b["cost_usd"]),
        "quality_change_pct": pct(m["quality_score"], b["quality_score"]),
        "depth_savings_pct": pct(m["depth_used"], b["depth_used"]) * -1 if b["depth_used"] else 0.0,
        "carbon_saved_g": round((b["carbon_g"] or 0.0) - (m["carbon_g"] or 0.0), 6),
        "energy_saved_kwh": round((b["energy_kwh"] or 0.0) - (m["energy_kwh"] or 0.0), 10),
    }