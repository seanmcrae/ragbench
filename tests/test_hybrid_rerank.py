import pytest

from rag_eval.datasets import Dataset, Document
from rag_eval.retrieval import BM25Retriever, ScoredDoc
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.embedders import HashingEmbedder
from rag_eval.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from rag_eval.retrieval.rerank import QueryLikelihoodReranker


def test_rrf_hand_computed() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    scores = {s.doc_id: s.score for s in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [s.doc_id for s in fused] == ["b", "c", "a", "d"]


def test_rrf_weights_and_tie_break() -> None:
    fused = reciprocal_rank_fusion([["x"], ["y"]], k=10)
    assert [s.doc_id for s in fused] == ["x", "y"]  # equal scores fall back to id order
    weighted = reciprocal_rank_fusion([["x"], ["y"]], k=10, weights=[1.0, 2.0])
    assert [s.doc_id for s in weighted] == ["y", "x"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["x"]], weights=[1.0, 2.0])


def test_hybrid_retriever_fuses_components(tiny_dataset: Dataset) -> None:
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(HashingEmbedder(dim=256))])
    hybrid.index(list(tiny_dataset.corpus.values()))
    hits = hybrid.search("What is the API rate limit on the Team plan?", k=3)
    assert hits[0].doc_id == "d1"
    assert len(hits) == 3
    assert hybrid.name == "hybrid[bm25+dense[hashing-256]]"
    assert hybrid.embedding_usage("q") == {}
    with pytest.raises(ValueError):
        HybridRetriever([BM25Retriever()])


def test_query_likelihood_prefers_phrase_match() -> None:
    # Same length and the same query unigrams; only the adjacent "rate limit" pair differs.
    docs = {
        "phrase": Document("phrase", "", "the rate limit is 300 per minute for api calls"),
        "scattered": Document("scattered", "", "limit api usage; the rate of calls varies widely"),
    }
    reranker = QueryLikelihoodReranker()
    reranker.fit(list(docs.values()))
    ranked = reranker.rerank(
        "api rate limit", [ScoredDoc("scattered", 9.0), ScoredDoc("phrase", 1.0)], docs
    )
    assert [s.doc_id for s in ranked] == ["phrase", "scattered"]
