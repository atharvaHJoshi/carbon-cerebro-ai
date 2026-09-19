from __future__ import annotations

from app.core.analysis import (
    analyze_conversation,
    analyze_prompt,
    complexity_features,
    detect_domain,
    detect_intent,
    required_context,
)
from app.tokenizer import count_tokens


def test_analyze_code_prompt(code_prompt):
    pi = analyze_prompt(code_prompt)
    assert pi.top_domain == "software"
    assert pi.intent in ("code_generation", "reasoning", "math_problem")
    assert pi.tokens > 0
    assert 0.0 <= pi.complexity <= 1.0
    assert pi.complexity_level in ("low", "medium", "high", "critical")


def test_analyze_question_prompt():
    pi = analyze_prompt("What is the capital of France?")
    assert pi.intent == "question_answering"
    assert pi.intent_confidence > 0.5
    assert pi.required_context < 0.5


def test_analyze_finance_domain():
    pi = analyze_prompt("I need to calculate the compound interest on my investment portfolio.")
    assert pi.top_domain in ("finance", "math")


def test_analyze_medical_domain():
    pi = analyze_prompt("What dosage of ibuprofen is safe for a patient with renal disease?")
    assert pi.top_domain == "medical"
    assert pi.domain_confidence > 0.3


def test_analyze_empty_returns_defaults():
    pi = analyze_prompt("")
    assert pi.intent != ""
    assert pi.tokens == 0
    assert pi.complexity >= 0.0


def test_detect_intent_code():
    intent, conf, reasons = detect_intent("Write a Python function that sorts a list")
    assert intent == "code_generation"
    assert conf > 0.0


def test_detect_domain_returns_scores():
    scores, top, conf, reasons = detect_domain("This is about debugging a python API connection error")
    assert top in scores
    assert conf > 0.0


def test_complexity_features_low_vs_high():
    low = complexity_features({"n_words": 5, "rare_ratio": 0.0, "math_count": 0, "code_indicators": False,
                               "questions": 0, "unique_ratio": 0.5, "avg_word_len": 4.0, "upper_ratio": 0.0},
                              "hello world this is short")
    high = complexity_features({"n_words": 300, "rare_ratio": 0.8, "math_count": 8, "code_indicators": True,
                                "questions": 3, "unique_ratio": 0.9, "avg_word_len": 9.0, "upper_ratio": 0.2},
                               "x" * 1000)
    assert high["score"] > low["score"]
    assert high["level"] in ("high", "critical")
    assert low["level"] in ("low", "medium")


def test_required_context_detects_references():
    s, reasons = required_context("As we discussed earlier, can you extend that idea?", "question_answering",
                                  {"n_words": 12})
    assert s > 0.4


def test_analyze_conversation_blends():
    messages = [
        {"role": "user", "content": "Explain recursion."},
        {"role": "assistant", "content": "Recursion is when a function calls itself."},
        {"role": "user", "content": "Give a python example."},
        {"role": "assistant", "content": "def factorial(n): return n * factorial(n-1)"},
        {"role": "user", "content": "What about tail recursion?"},
    ]
    pi = analyze_conversation(messages)
    assert pi.complexity >= 0.0
    assert pi.tokens > 0


def test_count_tokens_positive():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0