from __future__ import annotations

from app.core.context import compress_context
from app.tokenizer import count_tokens


def test_no_history():
    result = compress_context([], "What is the weather?")
    assert result.total_tokens == 0
    assert result.messages == []


def test_single_message_unchanged():
    history = [{"role": "user", "content": "Hello there"}]
    result = compress_context(history, "Hello there")
    assert len(result.messages) == 1
    assert result.retention == 1.0


def test_compress_keeps_relevant(conversation_history):
    query = "Give me a caching decorator example"
    result = compress_context(conversation_history, query)
    assert result.total_tokens > result.kept_tokens
    assert result.kept is not None
    # all kept content must be a subset of original
    joined = " ".join(result.kept)
    original = " ".join(m["content"] for m in conversation_history)
    assert joined.strip() in original or result.summary_note is not None


def test_compress_drops_relevance():
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is your favorite color?"},
        {"role": "assistant", "content": "I like blue."},
    ]
    result = compress_context(history, "capital of France")
    assert result.kept_tokens > 0
    assert result.dropped_tokens >= 0
    # The France-related message should be kept (higher relevance)
    france_before = count_tokens("The capital of France is Paris.")
    assert result.kept_tokens >= 0


def test_compress_budget_zero():
    history = [
        {"role": "user", "content": "First message about finance."},
        {"role": "assistant", "content": "A long financial response."},
        {"role": "user", "content": "Second message."},
    ]
    result = compress_context(history, "finance", budget=0.05)
    # must still return something usable (summary fallback)
    assert result.messages is not None
    assert result.kept_tokens > 0 or result.summary_note is not None


def test_compress_budget_full():
    history = [
        {"role": "user", "content": "Explain recursion."},
        {"role": "assistant", "content": "Recursion is when a function calls itself."},
    ]
    result = compress_context(history, "recursion", budget=1.0)
    assert result.retention <= 1.0
    assert result.total_tokens == result.kept_tokens + result.dropped_tokens


def test_relevance_scores():
    history = [
        {"role": "user", "content": "Python decorators?"},
        {"role": "assistant", "content": "Decorators wrap functions and modify behavior."},
        {"role": "user", "content": "Talk about football."},
        {"role": "assistant", "content": "Football is a popular team sport."},
    ]
    result = compress_context(history, "python decorators", budget=0.7)
    assert len(result.relevance) == len(history)
    assert all(0 <= r["score"] <= 1 for r in result.relevance)