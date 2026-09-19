"""Intelligent Context Compression (RQ2).

Compresses long conversations by:
  1. embedding every message against the latest user query,
  2. relevance scoring (cosine),
  3. redundancy removal via MMR (Maximal Marginal Relevance),
  4. sentence-level trimming when a budget is still exceeded,
  5. producing an extractive summary (medoids per cluster) as fallback.
"""

from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Sequence

from ..config import settings
from ..tokenizer import count_tokens
from .embeddings import cached_embed, cosine, embed_many

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Message:
    role: str
    content: str


@dataclass
class ContextResult:
    messages: list[Message]
    kept: list[str]
    dropped: list[str]
    relevance: list[dict]
    total_tokens: int
    kept_tokens: int
    dropped_tokens: int
    retention: float
    redundancy_removed: int
    summary_note: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "kept": self.kept,
            "dropped": self.dropped,
            "relevance": self.relevance,
            "total_tokens": self.total_tokens,
            "kept_tokens": self.kept_tokens,
            "dropped_tokens": self.dropped_tokens,
            "retention": round(self.retention, 4),
            "redundancy_removed": self.redundancy_removed,
            "summary_note": self.summary_note,
            "reasons": self.reasons,
        }


def _to_messages(history: Sequence[dict] | Sequence[Message]) -> list[Message]:
    out: list[Message] = []
    for m in history:
        if isinstance(m, dict):
            out.append(Message(role=m.get("role", "user"), content=m.get("content", "")))
        else:
            out.append(m)
    return out


def mmr_scores(query_v, message_vs: list, lam: float) -> list[int]:
    """Greedy MMR: pick messages maximising relevance minus redundancy to selection."""
    n = len(message_vs)
    sim_mat = [[cosine(message_vs[i], message_vs[j]) for j in range(n)] for i in range(n)]
    rel = [cosine(query_v, v) for v in message_vs]
    selected: list[int] = []
    remaining = list(range(n))
    while remaining:
        best = max(remaining, key=lambda i: (rel[i] - lam * max((sim_mat[i][j] for j in selected), default=0.0)))
        if rel[best] < 0.05 and selected:
            break
        selected.append(best)
        remaining.remove(best)
    return selected


def compress_context(history: Sequence[dict] | Sequence[Message], query: str,
                     budget: float | None = None, top_k: int | None = None,
                     lam: float | None = None) -> ContextResult:
    """Compress conversation history against the given query.

    `budget` is fractional token retention (0..1) or None for auto.
    """
    budget = settings.DEFAULT_CONTEXT_BUDGET if budget is None else budget
    top_k = settings.TOP_K_CONTEXT if top_k is None else top_k
    lam = settings.CONTEXT_REDUNDANCY_LAMBDA if lam is None else lam

    messages = _to_messages(history)
    if not messages:
        return ContextResult([], [], [], [], 0, 0, 0, 1.0, 0, None, ["No history provided"])
    if len(messages) <= 1:
        ms = messages
        tk = sum(count_tokens(m.content) for m in ms)
        return ContextResult(ms, [m.content for m in ms], [], _rel(messages, query), tk, tk, 0, 1.0, 0, None, ["Single-turn, no compression needed"])

    query_v = cached_embed(query)
    contents = [m.content for m in messages]
    vs = embed_many(contents).tolist()

    rel = [cosine(query_v, v) for v in vs]
    indices = mmr_scores(query_v, vs, lam)

    total_tok = sum(count_tokens(c) for c in contents)
    kept_tok = 0
    kept: list[int] = []
    budget_tok = max(int(total_tok * budget), 1)

    # keep messages ordered by conversation position for coherence
    order = sorted(indices)

    reasons: list[str] = []
    for idx in order:
        t = count_tokens(contents[idx])
        if kept_tok + t > budget_tok and kept:
            break
        if len(kept) >= top_k:
            break
        kept.append(idx)
        kept_tok += t

    # sentence-level trimming for the final kept messages if over budget
    if kept_tok > budget_tok and kept:
        _trim_keep(kept, contents, messages, vs, query_v, budget_tok, lam)

    reasons.append(f"Relevance filter kept {len(kept)}/{len(messages)} messages ({kept_tok}/{total_tok} tokens)")
    redundancy_removed = len(indices) - len(kept)
    if redundancy_removed > 0:
        reasons.append(f"MMR flagged {redundancy_removed} message(s) as redundant")

    summary_note = None
    kept_list = [content.strip() for i in kept for content in [messages[i].content]]
    dropped = [m.content for i, m in enumerate(messages) if i not in kept]
    kept_msgs = [messages[i] for i in kept]
    if not kept and indices:
        # nothing fit the budget: create an extractive summary message
        medoid = _cluster_medoid(vs, indices[:max(1, len(indices) // 2)])
        summary = "Conversation summary: " + " ".join(
            re.sub(r"\s+", " ", messages[i].content)[:400] for i in indices[: max(1, len(indices) // 2)]
        )[:1400]
        kept_msgs = [Message(role="assistant", content=summary)]
        dropped = [m.content for m in messages]
        kept_tok = count_tokens(summary)
        summary_note = "Budget exceeded even for a single message -> extractive summary emitted"
        reasons.append(summary_note)

    relevance_out = [{"idx": i, "role": m.role, "score": round(max(0.0, rel[i]), 4)} for i, m in enumerate(messages)]

    return ContextResult(
        messages=kept_msgs,
        kept=kept_list,
        dropped=dropped,
        relevance=relevance_out,
        total_tokens=total_tok,
        kept_tokens=kept_tok,
        dropped_tokens=total_tok - kept_tok,
        retention=kept_tok / max(1, total_tok),
        redundancy_removed=redundancy_removed,
        summary_note=summary_note,
        reasons=reasons,
    )


def _trim_keep(kept: list[int], contents: list[str], messages: list[Message], vs: list, query_v, budget_tok: int, lam: float) -> None:
    done = False
    for idx in list(kept):
        if done:
            break
        c = contents[idx]
        sents = _SENT_SPLIT_RE.split(c)
        if len(sents) < 2:
            continue
        sv = embed_many(sents).tolist()
        sel = mmr_scores(query_v, sv, lam)
        rebuilt = " ".join(sents[i] for i in sorted(sel))
        if count_tokens(rebuilt) < count_tokens(c):
            messages[idx].content = rebuilt
        if count_tokens(rebuilt) <= budget_tok:
            done = True


def _cluster_medoid(vs: list, indices: list[int]) -> int:
    best = indices[0]
    best_s = -1.0
    for i in indices:
        s = sum(cosine(vs[i], vs[j]) for j in indices)
        if s > best_s:
            best_s = s
            best = i
    return best


def _rel(messages: list[Message], query: str) -> list[dict]:
    query_v = cached_embed(query)
    return [{"idx": i, "role": m.role, "score": round(cosine(query_v, cached_embed(m.content)), 4)} for i, m in enumerate(messages)]