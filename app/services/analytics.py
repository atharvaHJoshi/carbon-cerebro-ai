"""Sustainability analytics - totals, per-model breakdown, time series,
and baseline-vs-GMA savings aggregation (Objective 9)."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func

from ..database import BenchmarkRun, RequestRecord, Session

_METRIC_KEYS = ["tokens_in", "tokens_out", "latency_ms", "energy_kwh", "carbon_kg", "cost_usd", "quality_score"]


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _sum(rows, key):
    return round(sum(r.get(key, 0.0) for r in rows if isinstance(r.get(key), (int, float))), 6)


def summary(db: Session) -> dict:
    rows = [r.to_dict() for r in db.query(RequestRecord).order_by(RequestRecord.id.desc()).limit(500).all()]
    gma = [r for r in rows if r["kind"] == "gma"]
    base = [r for r in rows if r["kind"] == "baseline"]

    totals = {
        "requests_total": len(rows),
        "gma_requests": len(gma),
        "baseline_requests": len(base),
        "total_energy_kwh": _sum(rows, "energy_kwh"),
        "total_carbon_kg": _sum(rows, "carbon_kg"),
        "total_cost_usd": _sum(rows, "cost_usd"),
        "total_tokens": _sum(rows, "tokens_in") + _sum(rows, "tokens_out"),
    }

    per_model: dict[str, dict] = defaultdict(lambda: {"count": 0, "carbon_kg": 0.0, "energy_kwh": 0.0,
                                                      "tokens": 0, "avg_latency_ms": 0.0, "avg_quality": 0.0})
    for r in rows:
        m = per_model[r["model_name"]]
        m["count"] += 1
        m["carbon_kg"] += r["carbon_kg"]
        m["energy_kwh"] += r["energy_kwh"]
        m["tokens"] += r["tokens_in"] + r["tokens_out"]
        m["avg_latency_ms"] += r["latency_ms"]
        m["avg_quality"] += r["quality_score"]
    for m in per_model.values():
        c = m["count"] or 1
        m["avg_latency_ms"] = round(m["avg_latency_ms"] / c, 2)
        m["avg_quality"] = round(m["avg_quality"] / c, 3)
    per_model = dict(sorted(per_model.items(), key=lambda kv: -kv[1]["carbon_kg"]))

    savings: dict | None = None
    if gma and base:
        savings = {
            "token_reduction_pct": round(_sum(gma, "tokens_in") / max(1, _sum(base, "tokens_in")) * 100.0, 2),
            "energy_reduction_pct": round((1 - _sum(gma, "energy_kwh") / max(1e-12, _sum(base, "energy_kwh"))) * 100.0, 2),
            "carbon_reduction_pct": round((1 - _sum(gma, "carbon_kg") / max(1e-12, _sum(base, "carbon_kg"))) * 100.0, 2),
            "carbon_saved_kg": round(_sum(base, "carbon_kg") - _sum(gma, "carbon_kg"), 6),
            "energy_saved_kwh": round(_sum(base, "energy_kwh") - _sum(gma, "energy_kwh"), 6),
            "latency_avg_gma_ms": _avg(gma, "latency_ms"),
            "latency_avg_baseline_ms": _avg(base, "latency_ms"),
            "quality_avg_gma": _avg(gma, "quality_score"),
            "quality_avg_baseline": _avg(base, "quality_score"),
            "avg_prune_ratio": _avg(gma, "prune_ratio"),
            "avg_context_ratio": _avg(gma, "context_ratio"),
        }

    return {"totals": totals, "per_model": per_model, "savings": savings, "compare": _compare_pairs(gma, base)}


def _compare_pairs(gma: list[dict], base: list[dict]) -> dict:
    """Pairwise average delta when each scenario has one baseline + one GMA run."""
    if not gma or not base:
        return {}
    pairs = {}
    for g in gma:
        key = (g["prompt"], g["model_name"])
        matches = [b for b in base if (b["prompt"], b["model_name"]) == key]
        for b in matches:
            pairs[key] = {
                "tokens_saved": b["tokens_in"] - g["tokens_in"],
                "latency_ms_saved": b["latency_ms"] - g["latency_ms"],
                "energy_saved_kwh": b["energy_kwh"] - g["energy_kwh"],
                "carbon_saved_kg": b["carbon_kg"] - g["carbon_kg"],
                "cost_saved_usd": b["cost_usd"] - g["cost_usd"],
                "quality_delta": g["quality_score"] - b["quality_score"],
            }
    if not pairs:
        return {}
    n = len(pairs)
    acc = {k: sum(p[k] for p in pairs.values()) / n for k in next(iter(pairs.values()))}
    acc["pairs"] = n
    return acc


def timeseries(db: Session, points: int = 20) -> list[dict]:
    rows = [r.to_dict() for r in db.query(RequestRecord).order_by(RequestRecord.id.asc()).all()]
    if not rows:
        return []
    step = max(1, len(rows) // points)
    out = []
    bucket: list[dict] = []
    for i, r in enumerate(rows):
        bucket.append(r)
        if (i + 1) % step == 0 or i == len(rows) - 1:
            ts = bucket[0]["created_at"][:16]
            out.append({
                "ts": ts,
                "gma_carbon_g": round(_sum([b for b in bucket if b["kind"] == "gma"], "carbon_kg") * 1000, 4),
                "baseline_carbon_g": round(_sum([b for b in bucket if b["kind"] == "baseline"], "carbon_kg") * 1000, 4),
                "gma_latency_ms": _avg([b for b in bucket if b["kind"] == "gma"], "latency_ms"),
                "baseline_latency_ms": _avg([b for b in bucket if b["kind"] == "baseline"], "latency_ms"),
                "count": len(bucket),
            })
            bucket = []
    return out


def benchmark_runs(db: Session, limit: int = 20) -> list[dict]:
    rows = db.query(BenchmarkRun).order_by(BenchmarkRun.id.desc()).limit(limit).all()
    return [r.to_dict() for r in rows]


def recent_requests(db: Session, limit: int = 50) -> list[dict]:
    rows = db.query(RequestRecord).order_by(RequestRecord.id.desc()).limit(limit).all()
    return [r.to_dict() for r in rows]