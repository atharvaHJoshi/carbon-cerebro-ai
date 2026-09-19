"""Adaptive LoRA selection (RQ3).

Maintains a catalog of domain-specific LoRA adapters. For an incoming query it:
  * blends the domain lexicon belief with keyword overlap and a semantic match
    against the adapter's prototype embedding,
  * ranks candidates and returns the best adapter with confidence,
  * falls back to the base model when no adapter clears the confidence floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import settings
from .embeddings import cached_embed, cosine

ADAPTER_CATALOG: list[dict] = [
    {"id": "lora-code",        "name": "Coding LoRA",         "domain": "software",
     "keywords": ["def ", "function", "api", "python", "bug", "debug", "refactor", "sql", "json", "docker", "algorithm"],
     "proto": "writing and debugging code, functions, apis, algorithms, database queries"},
    {"id": "lora-finance",     "name": "Finance LoRA",        "domain": "finance",
     "keywords": ["stock", "revenue", "tax", "investment", "balance sheet", "cash flow", "audit", "portfolio"],
     "proto": "stock market, investments, revenue, tax, banking, financial statements"},
    {"id": "lora-medical",     "name": "Medical LoRA",        "domain": "medical",
     "keywords": ["diagnosis", "symptom", "patient", "drug", "dosage", "clinical", "treatment"],
     "proto": "patient symptoms, diagnosis, treatment, drugs, clinical care"},
    {"id": "lora-legal",       "name": "Legal LoRA",          "domain": "legal",
     "keywords": ["contract", "clause", "lawsuit", "court", "compliance", "patent", "liability"],
     "proto": "contracts, courts, laws, litigation, compliance and legal clauses"},
    {"id": "lora-math",        "name": "Math/Science LoRA",   "domain": "math",
     "keywords": ["equation", "integral", "matrix", "probability", "proof", "theorem"],
     "proto": "equations, calculus, probability, proofs, statistics and scientific reasoning"},
    {"id": "lora-science",     "name": "Research LoRA",       "domain": "science",
     "keywords": ["experiment", "energy", "carbon", "climate", "quantum", "molecule"],
     "proto": "physics chemistry biology experiments, energy, climate, molecules"},
    {"id": "lora-general",     "name": "General Instruction LoRA", "domain": "general",
     "keywords": [],
     "proto": "everyday questions, general knowledge, clear helpful answers"},
]


@dataclass
class LoRADecision:
    selected: str | None
    confidence: float
    pipeline: list[dict]
    candidates: list[dict]
    fallback_to_base: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "confidence": round(self.confidence, 4),
            "pipeline": self.pipeline,
            "candidates": self.candidates,
            "fallback_to_base": self.fallback_to_base,
            "reasons": self.reasons,
        }


def _keyword_overlap(text: str, keywords: list[str]) -> float:
    low = text.lower()
    hits = sum(1 for k in keywords if k in low)
    denom = len(keywords) or 1
    return hits / denom


def select_adapter(query: str, domain_scores: dict[str, float] | None = None) -> LoRADecision:
    qv = cached_embed(query)
    candidates: list[dict] = []
    reasons: list[str] = []

    for ad in ADAPTER_CATALOG:
        domain_belief = (domain_scores or {}).get(ad["domain"], 0.0)
        semantic = cosine(qv, cached_embed(ad["proto"]))
        overlap = _keyword_overlap(query, ad["keywords"])
        score = 0.55 * min(domain_belief, 1.0) + 0.25 * semantic + 0.20 * overlap
        if ad["id"].endswith("general"):
            score += settings.LORA_BASE_REWARD
        candidates.append({"id": ad["id"], "name": ad["name"], "domain": ad["domain"],
                           "score": round(score, 4),
                           "parts": {"domain": round(domain_belief, 3),
                                     "semantic": round(semantic, 3),
                                     "keyword": round(overlap, 3)}})

    candidates.sort(key=lambda c: -c["score"])
    best = candidates[0]
    margin = best["score"] - candidates[1]["score"] if len(candidates) > 1 else 0.0
    confidence = min(1.0, best["score"] + margin * 0.4)

    fallback = confidence < settings.LORA_MIN_CONFIDENCE
    pipeline = []
    if fallback:
        selected = None
        pipeline.append({"step": "adapter-selection", "outcome": "fallback-to-base",
                         "detail": f"best adapter confidence {confidence:.2f} below floor {settings.LORA_MIN_CONFIDENCE}"})
        reasons.append(f"No adapter cleared confidence floor (best={best['id']} @ {confidence:.2f}); using base model only")
    else:
        selected = best["id"]
        pipeline.append({"step": "adapter-selection", "outcome": f"selected {best['name']}",
                         "detail": f"score {best['score']:.3f} | domain {best['parts']['domain']:.2f} semantic {best['parts']['semantic']:.2f} keyword {best['parts']['keyword']:.2f}"})
        reasons.append(f"Domain signals favour {best['name']} (confidence {confidence:.2f})")

    return LoRADecision(
        selected=selected,
        confidence=confidence,
        pipeline=[{"step": "adapter-selection", "outcome": "ok", "detail": "ranked catalog nov"}],
        candidates=candidates,
        fallback_to_base=fallback,
        reasons=reasons,
    )


def adapter_quality_boost(adapter_id: str | None, base_quality: float) -> float:
    """Quality uplift from applying a matching adapter."""
    if not adapter_id:
        return base_quality
    return min(base_quality + 0.08, 0.99)


def adapter_energy_overhead(adapter_id: str | None) -> float:
    """Relative energy overhead of loading/applying the adapter (~0 for base)."""
    return 0.03 if adapter_id else 0.0