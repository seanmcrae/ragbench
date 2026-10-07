import math

import pytest

from rag_eval.datasets import Document
from rag_eval.metrics import (
    exact_match,
    extract_citations,
    groundedness,
    latency_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    token_f1,
)

QRELS = {"a": 2, "b": 1, "c": 0, "z": 1}
RANKING = ["c", "a", "x", "b", "y"]


def test_recall_and_precision_hand_computed() -> None:
    assert recall_at_k(RANKING, QRELS, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKING, QRELS, 5) == pytest.approx(2 / 3)
    assert precision_at_k(RANKING, QRELS, 2) == pytest.approx(1 / 2)
    assert precision_at_k(RANKING, QRELS, 5) == pytest.approx(2 / 5)
    assert precision_at_k(["a"], QRELS, 10) == pytest.approx(1 / 10)
    assert recall_at_k(RANKING, {"c": 0}, 5) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(RANKING, QRELS, 0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKING, QRELS) == pytest.approx(1 / 2)
    assert reciprocal_rank(RANKING, QRELS, k=1) == 0.0
    assert reciprocal_rank(["x", "y"], QRELS) == 0.0


def test_ndcg_hand_computed() -> None:
    # gains at ranks 1..4 for c, a, x, b: 0, 3, 0, 1
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    # ideal gains: 3, 1, 1
    ideal = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKING, QRELS, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b", "z"], QRELS, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x"], {"x": 0}, 3) == 0.0


def test_retrieval_metrics_bundle_keys() -> None:
    out = retrieval_metrics(RANKING, QRELS, ks=[1, 5])
    assert set(out) == {
        "mrr",
        "recall@1",
        "precision@1",
        "ndcg@1",
        "recall@5",
        "precision@5",
        "ndcg@5",
    }


def test_exact_match_and_f1() -> None:
    assert exact_match("The Team plan.", ["team plan"]) == 1.0
    assert exact_match("Team", ["Business", "team"]) == 1.0
    assert exact_match("Team plan and above", ["Team"]) == 0.0
    # pred tokens: team plan and above (4); ref: team (1); overlap 1 -> P=1/4, R=1
    assert token_f1("Team plan and above", ["Team"]) == pytest.approx(2 * 0.25 / 1.25)
    assert token_f1("nothing shared", ["Team"]) == 0.0
    assert token_f1("", [""]) == 1.0
    assert token_f1("x", []) == 0.0


def test_citation_extraction() -> None:
    text = "It is 300 [d1]. Also see [d2, d1] and [d3]."
    assert extract_citations(text) == ["d1", "d2", "d3"]


def test_groundedness_heuristic() -> None:
    docs = {
        "d1": Document("d1", "Limits", "The API rate limit is 300 requests per minute."),
        "d2": Document("d2", "Other", "Unrelated text about exports."),
    }
    full = groundedness("The limit is 300 requests per minute [d1].", docs, {"d1", "d2"})
    assert full.citation_rate == 1.0
    assert full.support == 1.0
    assert full.score == 1.0
    # Second sentence uncited; "exports" not supported by the only cited doc.
    half = groundedness("Limit is 300 [d1]. Exports too.", docs, {"d1"})
    assert half.citation_rate == pytest.approx(0.5)
    assert half.support == pytest.approx(2 / 3)
    # Citing a document that was not retrieved does not count.
    assert groundedness("Limit is 300 [d2].", docs, {"d1"}).score == 0.0
    assert groundedness("", docs, {"d1"}).score == 0.0


def test_latency_summary() -> None:
    summary = latency_summary([10.0, 20.0, 30.0, 40.0, 100.0])
    assert summary["p50"] == pytest.approx(30.0)
    assert summary["p95"] == pytest.approx(88.0)
    assert latency_summary([]) == {"p50": 0.0, "p95": 0.0}
