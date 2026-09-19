"""Benchmark harness: runs a suite of scenarios through baseline and the GMA
pipeline, compares measured metrics and stores a comparative report."""

from __future__ import annotations

import json
import time

import numpy as np

from ..core.analysis import analyze_prompt
from ..core.evaluation import evaluate_response
from ..database import BenchmarkRun, Session
from ..schemas import BaselineRequest, GenerateRequest
from .baseline import run_baseline
from .orchestrator import compute_delta, run_generate


def benchmark_scenarios(scenarios: list[dict], *, baseline_model: str | None = None,
                        default_model: str = "llama-3-8b", name: str = "automated-benchmark",
                        region: str | None = None, save_records: bool = True,
                        db: Session | None = None) -> dict:
    if not scenarios:
        # default scenario suite derived from built-in prompts
        scenarios = [
            {"prompt": "Summarize the key ideas of sustainable AI in two sentences.", "reference": "Sustainable AI reduces unnecessary computation and environmental impact while preserving response quality."},
            {"prompt": "Write a Python function that computes the nth Fibonacci number efficiently."},
            {"prompt": "Explain what a LoRA adapter is and why it saves compute during fine-tuning."},
            {"prompt": "Classify the following review as positive or negative: 'The product works but the battery drains fast.'"},
            {"prompt": "Calculate the carbon footprint of 1 million LLM requests, assuming 0.005 kWh per request and a grid intensity of 0.5 kg CO2/kWh."},
            {"prompt": "Draft a short email to a client about a project delay."},
            {"prompt": "Translate 'Good morning, the meeting is at ten' into Spanish."},
            {"prompt": "List three best practices for reducing inference latency without losing quality."},
        ]

    results = []
    start = time.time()
    for i, sc in enumerate(scenarios):
        gen = GenerateRequest(
            prompt=sc["prompt"], history=sc.get("history"), max_tokens=sc.get("max_tokens", 128),
            reference=sc.get("reference"), region=region,
        )
        base = BaselineRequest(
            prompt=sc["prompt"], history=sc.get("history"), max_tokens=sc.get("max_tokens", 128),
            model=baseline_model or "llama-3-8b", reference=sc.get("reference"), region=region,
        )
        gma_out = run_generate(gen, db=db if save_records else None)
        base_out = run_baseline(base, db=db if save_records else None)
        delta = compute_delta(gma_out["result"]["structured_metrics"], base_out["metrics"])

        results.append({
            "scenario": i + 1,
            "prompt": sc["prompt"][:160],
            "gma": {
                "model": gma_out["execution_plan"]["model"],
                "adapter": gma_out["execution_plan"]["adapter"],
                "tokens_in": gma_out["result"]["structured_metrics"]["tokens_in"],
                "tokens_out": gma_out["result"]["structured_metrics"]["tokens_out"],
                "latency_ms": gma_out["result"]["structured_metrics"]["latency_ms"],
                "energy_kwh": gma_out["result"]["structured_metrics"]["energy_kwh"],
                "carbon_g": gma_out["result"]["structured_metrics"]["carbon_g"],
                "cost_usd": gma_out["result"]["structured_metrics"]["cost_usd"],
                "quality": gma_out["result"]["structured_metrics"]["quality_score"],
                "depth_used": gma_out["result"]["structured_metrics"]["depth_used"],
                "explanation": gma_out["explanation"]["heading"] if gma_out.get("explanation") else "",
            },
            "baseline": {
                "model": base_out["metrics"]["model"],
                "tokens_in": base_out["metrics"]["tokens_in"],
                "tokens_out": base_out["metrics"]["tokens_out"],
                "latency_ms": base_out["metrics"]["latency_ms"],
                "energy_kwh": base_out["metrics"]["energy_kwh"],
                "carbon_g": base_out["metrics"]["carbon_g"],
                "cost_usd": base_out["metrics"]["cost_usd"],
                "quality": base_out["metrics"]["quality_score"],
                "depth_used": 1.0,
            },
            "delta": delta,
        })

    summary = aggregate(results)
    record_id = None
    if db is not None:
        run = BenchmarkRun(name=name, scenarios=len(results),
                           results=json.dumps(results, default=str),
                           summary=json.dumps(summary, default=str))
        db.add(run)
        db.commit()
        db.refresh(run)
        record_id = run.id

    return {
        "record_id": record_id,
        "name": name,
        "elapsed_s": round(time.time() - start, 2),
        "scenarios": len(results),
        "results": results,
        "summary": summary,
    }


def aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    keys = ["tokens_in", "tokens_out", "latency_ms", "energy_kwh", "carbon_g", "cost_usd", "quality"]
    base_totals = {k: sum(r["baseline"][k] for r in results) for k in keys}
    gma_totals = {k: sum(r["gma"][k] for r in results) for k in keys}

    def pct(new, old):
        return round((new - old) / old * 100.0, 2) if old else 0.0

    return {
        "scenarios_run": len(results),
        "baseline": {k: round(v / len(results), 4) for k, v in base_totals.items()},
        "gma": {k: round(v / len(results), 4) for k, v in gma_totals.items()},
        "reductions_pct": {
            "tokens": pct(gma_totals["tokens_in"], base_totals["tokens_in"]),
            "latency": pct(gma_totals["latency_ms"], base_totals["latency_ms"]),
            "energy": pct(gma_totals["energy_kwh"], base_totals["energy_kwh"]),
            "carbon": pct(gma_totals["carbon_g"], base_totals["carbon_g"]),
            "cost": pct(gma_totals["cost_usd"], base_totals["cost_usd"]),
            "quality_delta": pct(gma_totals["quality"], base_totals["quality"]),
        },
        "carbon_saved_g": round(base_totals["carbon_g"] - gma_totals["carbon_g"], 4),
        "energy_saved_kwh": round(base_totals["energy_kwh"] - gma_totals["energy_kwh"], 8),
        "avg_quality_delta": round(gma_totals["quality"] / len(results) - base_totals["quality"] / len(results), 4),
        "model_usage": _counts(results),
    }


def _counts(results: list[dict]) -> dict:
    from collections import Counter
    models = Counter(r["gma"]["model"] for r in results)
    adapters = Counter(r["gma"]["adapter"] or "base" for r in results)
    return {"models": dict(models), "adapters": dict(adapters)}