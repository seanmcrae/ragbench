"""Exact (brute-force) cosine retrieval over embedder outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_eval.datasets import Document
from rag_eval.retrieval.base import ScoredDoc
from rag_eval.retrieval.embedders import Embedder, Matrix
from rag_eval.text import count_tokens


class DenseRetriever:
    """Exact inner-product search; corpora here are small enough that ANN would only add noise."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._doc_ids: list[str] = []
        self._matrix: Matrix = np.zeros((0, 0), dtype=np.float32)

    @property
    def name(self) -> str:
        return f"dense[{self.embedder.name}]"

    def index(self, corpus: Sequence[Document]) -> None:
        texts = [doc.full_text for doc in corpus]
        self._doc_ids = [doc.doc_id for doc in corpus]
        self.embedder.fit(texts)
        self._matrix = self.embedder.embed(texts)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        scores = self._matrix @ self.embedder.embed([query])[0]
        k = min(k, len(self._doc_ids))
        # Stable sort on negated scores keeps ties in corpus order, so results are deterministic.
        order = np.argsort(-scores, kind="stable")[:k]
        return [ScoredDoc(self._doc_ids[i], float(scores[i])) for i in order]

    def embedding_usage(self, query: str) -> dict[str, int]:
        model = self.embedder.price_as
        return {model: count_tokens(query)} if model else {}
