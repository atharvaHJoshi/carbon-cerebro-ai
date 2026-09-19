from __future__ import annotations

from app.core.evaluation import evaluate_response
from app.tokenizer import count_tokens


def test_evaluate_with_reference():
    pred = "Green AI reduces energy use and carbon emissions while keeping answer quality."
    ref = "Sustainable Artificial Intelligence lowers computational cost and environmental impact while preserving quality."
    ev = evaluate_response(pred, ref, latency_ms=123.4, tokens_in=20, tokens_out=25)
    assert 0.0 <= ev.quality_score <= 1.0
    assert ev.latency_ms == 123.4
    assert ev.tokens_in == 20
    assert ev.tokens_out == 25
    assert len(ev.notes) > 0


def test_evaluate_without_reference():
    ev = evaluate_response("The capital of France is Paris.", None)
    assert 0.0 <= ev.quality_score <= 1.0


def test_keyword_coverage_matches():
    pred = "Write a function to sort an array of numbers."
    ref = "Create a function that sorts numeric arrays."
    ev = evaluate_response(pred, ref)
    assert ev.keyword_coverage >= 0.0


def test_similar_response_high_score():
    a = "Solar panels convert sunlight into electrical energy."
    b = "Photovoltaic cells turn solar radiation into electricity."
    ev = evaluate_response(a, b)
    assert ev.similarity > 0.0


def test_dissimilar_response_lower_score():
    a = "Cats are independent pets."
    b = "The stock market rose three percent today."
    ev = evaluate_response(a, b)
    assert ev.similarity < 1.0


def test_empty_prediction():
    ev = evaluate_response("", "Some reference string")
    assert ev.quality_score >= 0.0


def test_to_dict_shape():
    ev = evaluate_response("hello world", "hello world again", tokens_in=3, tokens_out=4)
    d = ev.to_dict()
    assert d["quality_score"] >= 0
    assert d["similarity"] >= 0
    assert d["keyword_coverage"] >= 0
    assert d["length_ratio"] > 0