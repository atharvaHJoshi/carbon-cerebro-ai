"""Dynamic Token Pruning (RQ1).

Assigns an importance score to every input token/chunk and greedily drops the
lowest-value tokens up to a configurable budget, while protecting
instruction-like content, numbers, identifiers and other high-signal spans.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..config import settings
from ..tokenizer import count_tokens, join_chunks, split_word_chunks
from .analysis import STOPWORDS, _COMMON_WORDS, _CUE_TERMS

_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_CODE_TOKEN_RE = re.compile(r"[_a-zA-Z][a-zA-Z0-9_]*\(|->|=>|==|!=|[{}();:]")
_CODE_START_RE = re.compile(r"def\s|class\s|import\s|from\s|SELECT|WHERE|INSERT|```")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_CASE_RE = re.compile(r"\b[a-z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]+)+\b")


@dataclass
class PruneResult:
    original_text: str
    pruned_text: str
    original_tokens: int
    pruned_tokens: int
    removed_tokens: int
    retention: float                 # pruned/original fraction kept
    compression_ratio: float         # 1 - retention
    importance: dict[str, float] = field(default_factory=dict)
    protected: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    removed_spans: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text,
            "pruned_text": self.pruned_text,
            "original_tokens": self.original_tokens,
            "pruned_tokens": self.pruned_tokens,
            "removed_tokens": self.removed_tokens,
            "retention": round(self.retention, 4),
            "compression_ratio": round(self.compression_ratio, 4),
            "importance": self.importance,
            "protected": self.protected,
            "reasons": self.reasons,
            "removed_spans": self.removed_spans[:40],
        }


def _importance(chunk: str, i: int, n: int, words: list[str], code_block: bool) -> float:
    """Heuristic importance in 0..1."""
    w = chunk.strip(" \n\t")
    low = w.lower()
    score = 0.30
    reasons_local: list[str] = []

    # 1. stopword / low-information tokens
    if low.strip(string_punct()).rstrip() in STOPWORDS:
        score -= 0.28
    if low in _COMMON_WORDS and low not in _CUE_TERMS:
        score -= 0.12

    # 2. position - later tokens carry more weight for the instruction tail
    if n > 1:
        pos = i / (n - 1)
        score += 0.18 * pos
        score -= 0.12 * (1 - pos)

    # 3. cue / action words
    for cue, bonus in _CUE_TERMS.items():
        if cue in low:
            score += bonus
            reasons_local.append(f"cue:{cue}")
            break

    # 4. rare vocabulary
    clean = re.sub(r"[^a-z0-9]+", "", low)
    if clean and clean not in _COMMON_WORDS and clean not in STOPWORDS:
        score += 0.16

    # 5. entities / numbers
    if _ENTITY_RE.search(w) or _NUMBER_RE.search(w):
        score += 0.12

    # 6. code identifiers
    if _CODE_TOKEN_RE.search(w):
        score += 0.20
    if _SNAKE_CASE_RE.search(w) or _CAMEL_CASE_RE.search(w):
        score += 0.30
    if code_block:
        score += 0.25

    # 7. a token that is pure punctuation / whitespace noise
    if re.fullmatch(r"\s+", w):
        score -= 0.30
    if re.fullmatch(r"[,.;:!?'\"()]+", low.strip(" \n\t").lower()):
        score -= 0.10

    score = min(max(score, 0.0), 1.0)
    return score


def string_punct() -> str:
    return ".,;:!?\"'()[]{}<>"


def _is_instruction_line(chunk: str) -> bool:
    c = chunk.strip()
    if not c:
        return False
    return bool(re.match(r"^(you|you are|as an|role|task|instructions?|system|assistant|human|user|\s*>|#|```)", c))


def prune_prompt(prompt: str, max_ratio: float | None = None, protect_substrings: list[str] | None = None,
                 keep_ratio: float | None = None) -> PruneResult:
    """Drop low-importance input tokens within a removal budget."""
    max_ratio = settings.MAX_PRUNE_RATIO if max_ratio is None else max(max_ratio, 0.0)
    keep_ratio = settings.MIN_PRUNE_KEEP if keep_ratio is None else keep_ratio

    chunks = split_word_chunks(prompt)
    n = len(chunks)
    if n < 8:
        tok = count_tokens(prompt)
        return PruneResult(
            original_text=prompt, pruned_text=prompt,
            original_tokens=tok, pruned_tokens=tok, removed_tokens=0,
            retention=1.0, compression_ratio=0.0,
            importance={}, protected=[], reasons=["Prompt too short to prune safely"], removed_spans=[],
        )

    words = _WORD_LIST(prompt.lower())
    code_block = bool(_CODE_START_RE.search(prompt))

    scored: list[tuple[float, int]] = []
    important: dict[str, float] = {}
    for i, (_, _, chunk) in enumerate(chunks):
        im = _importance(chunk, i, n, words, code_block)
        scored.append((im, i))
        important[chunk.strip(" \n\t") or chunk] = round(im, 4)

    # protection: never drop spans the caller explicitly protects
    protected_indices: set[int] = set()
    protected_spans: list[str] = []
    if protect_substrings:
        for substr in protect_substrings:
            if substr and substr in prompt:
                start = prompt.find(substr)
                for idx, (s, e, chunk) in enumerate(chunks):
                    if s < start + len(substr) and e > start:
                        protected_indices.add(idx)
                        protected_spans.append(chunk.strip())
    # always protect instruction-like lines
    for idx, (_, _, chunk) in enumerate(chunks):
        if _is_instruction_line(chunk):
            protected_indices.add(idx)
            if chunk.strip():
                protected_spans.append(chunk.strip(" \n\t"))

    # sort ascending importance, skip protected
    to_remove: set[int] = set()
    budget = int(n * max_ratio)
    for _, idx in sorted(scored, key=lambda x: x[0]):
        if len(to_remove) >= budget:
            break
        idx = int(idx)
        if idx in protected_indices:
            continue
        to_remove.add(idx)

    # clamp by keep_ratio floor
    n_removed = len(to_remove)
    max_remove = n - max(1, int(round(n * keep_ratio)))
    if n_removed > max_remove:
        extra = n_removed - max_remove
        dropped = sorted(to_remove, key=lambda x: -important[chunks[x][2].strip(" \n\t") or chunks[x][2]])[:extra]
        to_remove -= set(dropped) if dropped else set()

    removed_spans = [chunks[i][2].strip() for i in sorted(to_remove) if chunks[i][2].strip()]

    kept = [c for i, c in enumerate(chunks) if i not in to_remove]
    if not to_remove:
        kept_text = prompt
    else:
        kept_text = join_chunks(kept)
        # tidy double spaces left behind by removed chunks
        kept_text = re.sub(r"[ \t]{2,}", " ", kept_text).strip()

    reasons = [
        f"Pruned {len(to_remove)} of {n} chunks ({len(to_remove) / max(1, n) * 100:.0f}% budget applied)",
        "Protected " + (", ".join(protected_spans[:5]) if protected_spans else "none") + " high-value spans",
        "Stopwords/punctuation penalised, cue-words & rare terms boosted",
    ]

    return PruneResult(
        original_text=prompt,
        pruned_text=kept_text,
        original_tokens=count_tokens(prompt),
        pruned_tokens=count_tokens(kept_text),
        removed_tokens=count_tokens(prompt) - count_tokens(kept_text),
        retention=count_tokens(kept_text) / max(1, count_tokens(prompt)),
        compression_ratio=1 - count_tokens(kept_text) / max(1, count_tokens(prompt)),
        importance=important,
        protected=protected_spans,
        reasons=reasons,
        removed_spans=removed_spans,
    )


import functools


@functools.lru_cache(maxsize=8)
def _WORD_LIST(text: str) -> list[str]:
    import re as _re
    return _re.findall(r"\b[\w’']+\b", text)