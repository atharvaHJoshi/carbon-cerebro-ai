from __future__ import annotations

from app.core.embeddings import cached_embed, cosine, embed_many, embed_text


def test_cosine_similar():
    a = embed_text("the quick brown fox jumps over the lazy dog")
    b = embed_text("a fast brown fox leaps above the idle canine")
    assert cosine(a, b) > 0.3


def test_cosine_dissimilar():
    a = embed_text("stock market investment banking revenue")
    b = embed_text("recursion data structure binary tree algorithm")
    assert cosine(a, b) < 0.5


def test_cosine_identical():
    a = embed_text("python programming for beginners")
    assert abs(cosine(a, a) - 1.0) < 1e-6


def test_cosine_empty():
    assert cosine(embed_text(""), embed_text("anything")) == 0.0


def test_embed_many_shape():
    arr = embed_many(["one", "two words here", "another"])
    assert arr.shape[0] == 3
    assert arr.shape[1] == 384


def test_cached_embed_is_stable():
    e1 = cached_embed("deterministic embeddings please")
    e2 = cached_embed("deterministic embeddings please")
    assert (e1 == e2).all()


def test_cosine_none_safe():
    assert cosine(None, None) == 0.0
    assert cosine(None, embed_text("x")) == 0.0