"""Multi-objective model & configuration router.

Scores every candidate execution strategy against the request's objectives
(quality, carbon, energy, cost, latency, resource use) with user-suppliable
weights, hard constraints and explanation. Also produces an Explainability
record (Objective 8) describing *why* a plan won.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings
from ..core.carbon import ModelEnergyProfile, estimate_carbon, energy_for_tokens_constants
from ..tokenizer import count_tokens
from .analysis import PromptIntelligence


@dataclass
class RouterDecision:
    request_id: str | None
    prompt_intelligence: dict
    candidates: list[dict]
    selected: dict
    runner_up: dict | None
    explanation: dict
    constraints: dict
    weights: dict
    wins: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "candidates": self.candidates,
            "selected": self.selected,
            "runner_up": self.runner_up,
            "explanation": self.explanation,
            "constraints": self.constraints,
            "weights": self.weights,
        }


def candidates_for_request(prompt_intelligence: PromptIntelligence,
                           adapter_id: str | None, layers_ratio: float,
                           prune_ratio: float, context_ratio: float,
                           tokens_in: int, tokens_out_estimate: int,
                           profile_overrides: dict[str, ModelEnergyProfile] | None = None) -> list[dict]:
    """Enumerate candidate (model, adapter, config) strategies."""
    pi = prompt_intelligence
    quality_capability = {
        "tinygpt": 0.35, "flan-t5-base": 0.55, "gemma-2b": 0.78, "llama-3-8b": 0.9,
        "mistral-7b": 0.88, "qwen-7b": 0.9, "gpt-large-70b": 0.98, "gemini-pro": 0.97,
        "claude-3.5": 0.98,
    }
    cost_per_token = {
        "tinygpt": 0.0, "flan-t5-base": 1.0e-7, "gemma-2b": 3.0e-7, "llama-3-8b": 8.0e-7,
        "mistral-7b": 7.0e-7, "qwen-7b": 8.0e-7, "gpt-large-70b": 5.0e-6, "gemini-pro": 3.0e-6,
        "claude-3.5": 3.0e-6,
    }
    latency_overhead = {"tinygpt": 30, "flan-t5-base": 80, "gemma-2b": 120, "llama-3-8b": 200,
                        "mistral-7b": 180, "qwen-7b": 200, "gpt-large-70b": 500, "gemini-pro": 450,
                        "claude-3.5": 420}
    resource_weight = {"tinygpt": 0.05, "flan-t5-base": 0.1, "gemma-2b": 0.25, "llama-3-8b": 0.5,
                       "mistral-7b": 0.45, "qwen-7b": 0.5, "gpt-large-70b": 0.95, "gemini-pro": 1.0,
                       "claude-3.5": 1.0}

    # quality multipliers
    diff_penalty = 0.18 * pi.complexity
    task_fit = {"summarization": 0.9, "classification": 0.9, "question_answering": 0.85,
                "code_generation": 0.8, "generation": 0.9, "translation": 0.9,
                "extraction": 0.85, "reasoning": 0.7, "editing": 0.9, "planning": 0.8,
                "recommendation": 0.85, "prediction": 0.8, "math_problem": 0.7}
    adapter_boost = 0.08 if adapter_id else 0.0

    candidates: list[dict] = []
    profiles = profile_overrides or {}
    for mid in quality_capability:
        profile = profiles.get(mid) or energy_for_tokens_constants(mid)
        base_q = quality_capability[mid] * task_fit.get(pi.intent, 0.85)
        quality = max(0.0, base_q - diff_penalty + adapter_boost) * layers_ratio

        e_est = estimate_carbon(profile, tokens_in, tokens_out_estimate,
                                latency_ms=latency_overhead[mid] * 1.0,
                                layers_ratio=layers_ratio)
        latency = (tokens_in / profile.prefill_tps + tokens_out_estimate / profile.decode_tps) * 1000.0
        latency += latency_overhead[mid] * (0.5 + 0.5 * layers_ratio)
        if adapter_id:
            latency *= 1.03

        cost = (tokens_in + tokens_out_estimate) * cost_per_token[mid]
        candidates.append({
            "model": mid,
            "profile": profile.to_dict(),
            "adapter": adapter_id,
            "layers_ratio": layers_ratio,
            "prune_ratio": prune_ratio,
            "context_ratio": context_ratio,
            "quality": round(min(quality, 0.99), 4),
            "energy_kwh": round(e_est.energy_kwh, 8),
            "carbon_kg": round(e_est.carbon_kg, 8),
            "carbon_g": round(e_est.carbon_kg * 1000, 6),
            "latency_ms": round(latency, 1),
            "cost_usd": round(cost, 8),
            "resource": resource_weight[mid],
            "grid_region": e_est.grid_region,
        })
    return candidates


def _normalise(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo == 0:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _constraint_filter(candidates: list[dict], constraints: dict) -> list[dict]:
    out = []
    for c in candidates:
        if constraints.get("min_quality") and c["quality"] < constraints["min_quality"]:
            continue
        if constraints.get("max_carbon_g") and c["carbon_g"] > constraints["max_carbon_g"]:
            continue
        if constraints.get("max_latency_ms") and c["latency_ms"] > constraints["max_latency_ms"]:
            continue
        if constraints.get("max_cost_usd") and c["cost_usd"] > constraints["max_cost_usd"]:
            continue
        if constraints.get("exclude_models") and c["model"] in constraints["exclude_models"]:
            continue
        out.append(c)
    return out


def route(prompt_intelligence: PromptIntelligence, adapter_id: str | None, layers_ratio: float,
          prune_ratio: float, context_ratio: float, tokens_in: int, tokens_out_estimate: int,
          weights: dict | None = None, constraints: dict | None = None,
          profile_overrides: dict | None = None, request_id: str | None = None) -> RouterDecision:
    w = {**settings.DEFAULT_WEIGHTS, **(weights or {})}
    cons = {**{"min_quality": settings.MIN_REQUIRED_QUALITY}, **(constraints or {})}

    cands = candidates_for_request(prompt_intelligence, adapter_id, layers_ratio,
                                   prune_ratio, context_ratio, tokens_in, tokens_out_estimate,
                                   profile_overrides)
    filtered = _constraint_filter(cands, cons)

    if not filtered:
        # relax constraints and keep one fallback
        filtered = cands
        relaxed = True
    else:
        relaxed = False

    scores_q = _normalise([c["quality"] for c in filtered])
    scores_c = _normalise([-c["carbon_g"] for c in filtered])
    scores_e = _normalise([-c["energy_kwh"] for c in filtered])
    scores_cost = _normalise([-c["cost_usd"] for c in filtered])
    scores_lat = _normalise([-c["latency_ms"] for c in filtered])
    scores_res = _normalise([-c["resource"] for c in filtered])

    for i, c in enumerate(filtered):
        c["dim_scores"] = {
            "quality": round(scores_q[i], 4),
            "carbon": round(scores_c[i], 4),
            "energy": round(scores_e[i], 4),
            "cost": round(scores_cost[i], 4),
            "latency": round(scores_lat[i], 4),
            "resource": round(scores_res[i], 4),
        }
        total = (w["quality"] * scores_q[i] + w["carbon"] * scores_c[i] + w["energy"] * scores_e[i]
                 + w["cost"] * scores_cost[i] + w["latency"] * scores_lat[i] + w["resource"] * scores_res[i])
        c["score"] = round(total, 4)

    filtered.sort(key=lambda c: -c["score"])
    selected = filtered[0]
    runner = filtered[1] if len(filtered) > 1 else None

    explanation = build_explanation(selected, runner, cons, w, prompt_intelligence,
                                    adapter_id, relaxed, filtered)

    return RouterDecision(
        request_id=request_id,
        prompt_intelligence=prompt_intelligence.to_dict(),
        candidates=filtered,
        selected=selected,
        runner_up=runner,
        explanation=explanation,
        constraints=cons,
        weights={k: round(v, 3) for k, v in w.items()},
    )


def build_explanation(selected: dict, runner: dict | None, constraints: dict, weights: dict,
                      pi: PromptIntelligence, adapter_id: str | None, relaxed: bool,
                      all_candidates: list[dict]) -> dict:
    reasons = []
    reasons.append(f"Task complexity: {pi.complexity_level} ({pi.complexity:.2f})")
    reasons.append(f"Domain: {pi.top_domain.title()} | intent: {pi.intent}")
    reasons.append(f"Required context: {'High' if pi.required_context > 0.5 else 'Low'} ({pi.required_context:.2f})")
    reasons.append(f"Reasoning required: {pi.reasoning_required:.2f}")
    reasons.append(f"Estimated quality: {selected['quality'] * 100:.0f}%")
    reasons.append(f"Estimated latency: {selected['latency_ms']:.0f} ms")
    reasons.append(f"Estimated carbon: {selected['carbon_g'] * 1000:.2f} mg CO2")
    reasons.append(f"LoRA adapter: {adapter_id or 'base (no adapter, low confidence)'}")
    reasons.append(f"Layer plan: {selected['layers_ratio'] * 100:.0f}% of model depth")
    reasons.append(f"Input optimisation: {selected['prune_ratio'] * 100:.0f}% token pruning, {selected['context_ratio'] * 100:.0f}% context retention")
    if runner:
        gap = (selected["score"] - runner["score"]) / max(1e-9, runner["score"])
        reasons.append(f"Beat runner-up {runner['model']} by {gap * 100:.1f}% on weighted score")
    dom_s = selected.get("dim_scores", {})
    reasons.append(f"Dimension scores -> {dom_s}")
    if runner and selected["carbon_g"] < runner["carbon_g"]:
        reasons.append(f"Compared to {runner['model']}: {((1 - selected['carbon_g'] / max(1e-12, runner['carbon_g'])) * 100):.0f}% less carbon")
    if relaxed:
        reasons.append("NOTE: hard constraints were relaxed because no candidate satisfied all of them")

    return {
        "heading": f"Selected model: {selected['model']}",
        "reasons": reasons,
        "decision_path": [
            {"step": "prompt-analysis", "detail": f"{pi.intent}/{pi.top_domain}, complexity {pi.complexity:.2f}"},
            {"step": "token-pruning", "detail": f"{selected['prune_ratio'] * 100:.0f}% input reduction"},
            {"step": "context-compression", "detail": f"{selected['context_ratio'] * 100:.0f}% history retained"},
            {"step": "adapter-routing", "detail": adapter_id or "base"},
            {"step": "layer-plan", "detail": f"{selected['layers_ratio'] * 100:.0f}% depth"},
            {"step": "carbon-prediction", "detail": f"{selected['carbon_g'] * 1000:.2f} mg CO2"},
            {"step": "decision", "detail": f"{selected['model']} scored {selected['score']}"},
        ],
        "constraints": constraints,
        "weights": weights,
        "objective_scores": {k: v for k, v in dom_s.items()} if dom_s else {},
        "alternatives": [c["model"] for c in all_candidates[:3]],
    }


def model_catalog_overview() -> list[dict]:
    profiles = {}
    for mid in ["tinygpt", "flan-t5-base", "gemma-2b", "llama-3-8b", "mistral-7b", "gpt-large-70b"]:
        p = energy_for_tokens_constants(mid)
        profiles[mid] = p.to_dict()
    return [{"model": mid, "profile": pr} for mid, pr in profiles.items()]