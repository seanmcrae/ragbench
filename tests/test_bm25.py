import math

import pytest

from rag_eval.datasets import Dataset, Document
from rag_eval.retrieval import BM25Retriever


def test_hand_computed_score() -> None:
    # Three docs, one query term. Doc lengths are counted after stopword removal.
    docs = [
        Document("a", "", "apple apple banana"),
        Document("b", "", "banana cherry"),
        Document("c", "", "cherry cherry cherry date"),
    ]
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.index(docs)
    avg_len = (3 + 2 + 4) / 3
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / avg_len)
    expected = idf * 2 * 2.2 / (2 + norm)
    scores = bm25.score("apple")
    assert scores == {"a": pytest.approx(expected)}


def test_search_ranks_exact_matches_first(tiny_dataset: Dataset) -> None:
    bm25 = BM25Retriever()
    bm25.index(list(tiny_dataset.corpus.values()))
    hits = bm25.search("Slack sync interval", k=2)
    assert hits[0].doc_id == "d5"
    assert len(hits) == 1  # only one document shares any term
    assert bm25.search("unrelated words", k=3) == []
    assert bm25.embedding_usage("anything") == {}
