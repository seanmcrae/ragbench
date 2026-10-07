from collections.abc import Callable

import numpy as np
import pytest

from rag_eval.datasets import Dataset, Document
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.embedders import (
    Embedder,
    HashingEmbedder,
    TfidfSvdEmbedder,
    l2_normalize,
)


def test_l2_normalize_handles_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert out[0] == pytest.approx([0.6, 0.8])
    assert out[1] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("embedder_factory", [lambda: TfidfSvdEmbedder(dim=4), HashingEmbedder])
def test_dense_retrieval_finds_topical_document(
    tiny_dataset: Dataset, embedder_factory: Callable[[], Embedder]
) -> None:
    retriever = DenseRetriever(embedder_factory())
    retriever.index(list(tiny_dataset.corpus.values()))
    hits = retriever.search("export data as CSV", k=3)
    assert hits[0].doc_id == "d4"
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_tfidf_svd_is_deterministic(tiny_dataset: Dataset) -> None:
    texts = [d.full_text for d in tiny_dataset.corpus.values()]
    first, second = TfidfSvdEmbedder(dim=3, seed=1), TfidfSvdEmbedder(dim=3, seed=1)
    first.fit(texts)
    second.fit(texts)
    np.testing.assert_allclose(first.embed(texts), second.embed(texts))


def test_unfitted_embedder_raises() -> None:
    with pytest.raises(RuntimeError):
        TfidfSvdEmbedder().embed(["x"])


def test_offline_embedders_are_free() -> None:
    retriever = DenseRetriever(HashingEmbedder(dim=64))
    retriever.index([Document("a", "", "text")])
    assert retriever.embedding_usage("query") == {}
