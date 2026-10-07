"""Second-stage rerankers applied to the head of a first-stage ranking."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Protocol

from rag_eval.datasets import Document
from rag_eval.retrieval.base import ScoredDoc
from rag_eval.text import content_terms


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def fit(self, corpus: Sequence[Document]) -> None: ...

    def rerank(
        self, query: str, candidates: Sequence[ScoredDoc], docs: Mapping[str, Document]
    ) -> list[ScoredDoc]:
        """Return ``candidates`` re-ordered best first."""


class QueryLikelihoodReranker:
    """Dirichlet-smoothed query likelihood with an ordered-bigram component.

    A light version of the sequential dependence model (Metzler & Croft, 2005): adjacent query
    term pairs that also appear adjacently in a document earn extra weight, which rewards
    passages that state the queried phrase rather than merely mentioning its words.
    """

    def __init__(self, mu: float = 100.0, bigram_weight: float = 0.15) -> None:
        self.mu = mu
        self.bigram_weight = bigram_weight
        self._unigram_cf: Counter[str] = Counter()
        self._bigram_cf: Counter[tuple[str, str]] = Counter()
        self._collection_len = 0

    @property
    def name(self) -> str:
        return "ql-sdm"

    def fit(self, corpus: Sequence[Document]) -> None:
        self._unigram_cf = Counter()
        self._bigram_cf = Counter()
        for doc in corpus:
            terms = content_terms(doc.full_text)
            self._unigram_cf.update(terms)
            self._bigram_cf.update(pairwise(terms))
        self._collection_len = sum(self._unigram_cf.values())

    def _log_likelihood(self, tf: int, cf: int, doc_len: int) -> float:
        # Unseen-in-collection features get a half-count floor so the log stays finite.
        p_collection = max(cf, 0.5) / max(self._collection_len, 1)
        return math.log((tf + self.mu * p_collection) / (doc_len + self.mu))

    def score(self, query: str, doc: Document) -> float:
        q_terms = content_terms(query)
        d_terms = content_terms(doc.full_text)
        unigrams = Counter(d_terms)
        bigrams = Counter(pairwise(d_terms))
        uni = sum(
            self._log_likelihood(unigrams[t], self._unigram_cf[t], len(d_terms)) for t in q_terms
        )
        pairs = list(pairwise(q_terms))
        bi = sum(self._log_likelihood(bigrams[p], self._bigram_cf[p], len(d_terms)) for p in pairs)
        return (1 - self.bigram_weight) * uni + self.bigram_weight * bi

    def rerank(
        self, query: str, candidates: Sequence[ScoredDoc], docs: Mapping[str, Document]
    ) -> list[ScoredDoc]:
        rescored = [ScoredDoc(c.doc_id, self.score(query, docs[c.doc_id])) for c in candidates]
        return sorted(rescored, key=lambda s: (-s.score, s.doc_id))


class CrossEncoderReranker:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model
        self._model = CrossEncoder(model)

    @property
    def name(self) -> str:
        return f"cross-encoder:{self.model_name}"

    def fit(self, corpus: Sequence[Document]) -> None:
        return None

    def rerank(
        self, query: str, candidates: Sequence[ScoredDoc], docs: Mapping[str, Document]
    ) -> list[ScoredDoc]:
        pairs = [(query, docs[c.doc_id].full_text) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        rescored = [ScoredDoc(c.doc_id, float(s)) for c, s in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda s: (-s.score, s.doc_id))
