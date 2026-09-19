"""Token counting that uses tiktoken when available and falls back to a
calibrated estimators otherwise, so the platform works fully offline."""

from __future__ import annotations

import re
from functools import lru_cache

try:  # optional dependency; graceful degradation when unavailable
    import tiktoken  # type: ignore

    _TIKTOKEN_ENC = None

    def _get_enc():
        global _TIKTOKEN_ENC
        if _TIKTOKEN_ENC is None:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        return _TIKTOKEN_ENC

    TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover - environment without tiktoken
    TIKTOKEN_AVAILABLE = False
    _get_enc = None  # type: ignore


_WORD_RE = re.compile(r"\b[\w’']+\b", re.UNICODE)
_PUNCT_RE = re.compile(r"\s*\S+\s*")


def _fallback_tokens(text: str) -> int:
    """Deterministic approximation of cl100k token counts.

    Words contribute ~1.3 tokens; punctuation and spacing add a fixed per-word
    overhead. Calibrated against tiktoken on sample text.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return max(1, len(text) // 4)
    punct = len(re.findall(r"[^\w\s]", text))
    return max(1, int(len(words) * 1.25 + punct * 0.45 + 2))


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    try:
        if TIKTOKEN_AVAILABLE:
            return len(_get_enc().encode(text))
    except Exception:
        pass
    return _fallback_tokens(text)


def tokenize(text: str | None) -> list[str]:
    """Split into sub-word/word tokens for importance analysis."""
    if not text:
        return []
    if TIKTOKEN_AVAILABLE:
        try:
            enc = _get_enc()
            out = []
            for token in enc.encode(text):
                out.append(enc.decode_single_token_bytes(token).decode("utf-8", errors="replace"))
            return out
        except Exception:
            pass
    # fallback: whitespace + punctuation aware chunks
    return re.findall(r"\S+", text)


@lru_cache(maxsize=16)
def token_stats(text: str) -> dict:
    return {
        "tokens": count_tokens(text),
        "words": len(_WORD_RE.findall(text)),
        "chars": len(text),
        "method": "tiktoken" if TIKTOKEN_AVAILABLE else "fallback",
    }


def split_word_chunks(text: str) -> list[tuple[int, int, str]]:
    """Whitespace-aware chunks preserving original spans: (start, end, chunk)."""
    chunks: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\S+\s*", text):
        chunks.append((m.start(), m.end(), m.group()))
    return chunks


def join_chunks(chunks: list[tuple[int, int, str]]) -> str:
    return "".join(c[2] for c in chunks)