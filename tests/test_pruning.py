from __future__ import annotations

from app.core.pruning import prune_prompt


def test_prune_long_prompt():
    prompt = (
        "Please explain the concept of recursion in computer science. "
        "Recursion is when a function calls itself directly or indirectly. "
        "It is an important concept used in algorithms like binary search, "
        "tree traversal, and the quick sort algorithm. Give me a clear and "
        "detailed explanation with examples using Python programming language."
    )
    result = prune_prompt(prompt)
    assert result.original_tokens > 0
    assert result.pruned_tokens <= result.original_tokens
    assert 0.0 <= result.retention <= 1.0
    assert 0.0 <= result.compression_ratio <= 1.0
    assert result.pruned_text != ""


def test_prune_short_prompt_is_identity():
    prompt = "What is 2 + 2?"
    result = prune_prompt(prompt)
    assert result.original_text == prompt
    assert result.compression_ratio == 0.0
    assert result.retention == 1.0


def test_prune_protects_key_tokens():
    prompt = (
        "Write a function in Python named binary_search that takes a sorted "
        "array and a target value and returns the index. Please provide code "
        "and test it against edge cases."
    )
    result = prune_prompt(prompt)
    # key instruction kept
    assert "binary_search" in result.pruned_text or result.original_text == result.pruned_text


def test_prune_always_returns_nonempty():
    result = prune_prompt("A" * 200)
    assert result.pruned_text.strip() != ""


def test_prune_zero_ratio_keeps_all():
    prompt = "This is a moderately long prompt " * 5
    result = prune_prompt(prompt, max_ratio=0.0)
    assert result.compression_ratio == 0.0


def test_prune_retention_floor():
    prompt = ("word " * 50)
    result = prune_prompt(prompt, max_ratio=0.6)
    # keep floor should hold ~55%+ of chunks
    assert result.retention >= 0.5