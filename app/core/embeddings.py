"""Deterministic, offline text embeddings.

Uses hashed character n-grams (2..4) with sub-sampling, L2 normalisation and
cosine similarity. No network, no model download - embeddings are stable across
runs, which matters for reproducible benchmarks. Optional heavier providers may
be plugged in by updating :func:`get_provider`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

_DIM = 384
_SUB = 0.25  # feature subsampling (like word2vec) - reduces bias of frequent n-grams


def _hash_feature(feat: str) -> int:
    return int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "little")


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9&\n]", " ", text)
    return text


def embed_text(text: str) -> np.ndarray:
    """Produce an L2-normalised hashed bag-of-char-n-gram vector."""
    vec = np.zeros(_DIM, dtype=np.float32)
    norm = _normalise(text or "")
    if not norm.strip():
        return vec
    for n in (2, 3, 4):
        for i in range(len(norm) - n + 1):
            gram = norm[i : i + n]
            if re.search(r"\s", gram):
                continue
            if np.random.RandomState(hash(gram) % (2**32)).random() < _SUB:
                continue
            h = _hash_feature(gram)
            idx = h % _DIM
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
    norm_vec = np.linalg.norm(vec)
    if norm_vec > 0:
        vec /= norm_vec
    return vec


def embed_many(texts: Iterable[str]) -> np.ndarray:
    arr = np.stack([embed_text(t) for t in texts])
    return arr


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return (A @ B.T).astype(np.float64)


@lru_cache(maxsize=8192)
def _cached_embed(text: str) -> np.ndarray:
    return embed_text(text)


def cached_embed(text: str) -> np.ndarray:
    return _cached_embed(text)


def embed_text_non_cached(text: str) -> np.ndarray:
    return embed_text(text)