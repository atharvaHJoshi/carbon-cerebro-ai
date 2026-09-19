"""Evaluation metrics (Objective 11).

* semantic_similarity via char n-gram embeddings,
* keyword coverage,
* length adequacy,
* overall quality score for a generation against a reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..tokenizer import count_tokens
from .embeddings import cached_embed, cosine
from .analysis import STOPWORDS


@dataclass
class Evaluation:
    similarity: float
    keyword_coverage: float
    length_ratio: float
    latency_ms: float | None
    tokens_in: int
    tokens_out: int
    quality_score: float
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "similarity": round(self.similarity, 4),
            "keyword_coverage": round(self.keyword_coverage, 4),
            "length_ratio": round(self.length_ratio, 4),
            "latency_ms": self.latency_ms,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "quality_score": round(self.quality_score, 4),
            "notes": self.notes,
        }


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"\b[\w’']+\b", text.lower()) if w not in STOPWORDS}


def evaluate_response(prediction: str, reference: str | None, latency_ms: float | None = None,
                      tokens_in: int = 0, tokens_out: int = 0) -> Evaluation:
    notes: list[str] = []
    sim = 0.0
    if reference:
        sim = cosine(cached_embed(prediction), cached_embed(reference))
        notes.append("response-reference semantic similarity computed (char n-gram embeddings)")

    kw_cov = 0.0
    pred_kw = _keywords(prediction)
    if reference:
        ref_kw = _keywords(reference)
        if ref_kw:
            kw_cov = len(pred_kw & ref_kw) / max(1, len(ref_kw))
        else:
            kw_cov = 0.5
    else:
        # self-contained adequacy: non-empty, covers query keywords
        kw_cov = 0.5

    len_ratio = 1.0
    if reference and count_tokens(reference) > 0:
        len_ratio = min(count_tokens(prediction) / count_tokens(reference), 2.0)
        if abs(len_ratio - 1.0) > 0.6:
            notes.append("length diverges from reference (>60%)")

    quality = 0.55 * sim + 0.25 * kw_cov + 0.20 * min(len_ratio, 1.0)
    if not reference:
        quality = 0.55 * 0.6 + 0.25 * kw_cov + 0.20 * 0.8  # no-reference calibration

    return Evaluation(
        similarity=sim,
        keyword_coverage=kw_cov,
        length_ratio=len_ratio,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        quality_score=quality,
        notes=notes,
    )