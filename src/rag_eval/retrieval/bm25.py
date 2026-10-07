"""Okapi BM25 over an in-memory inverted index."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from rag_eval.datasets import Document
from rag_eval.retrieval.base import ScoredDoc, top_k
from rag_eval.text import content_terms


class BM25Retriever:
    """BM25 with the Lucene-style non-negative IDF: ln(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, list[tuple[str, int]]] = {}
        self._doc_len: dict[str, int] = {}
        self._avg_len = 0.0
        self._idf: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "bm25"

    def index(self, corpus: Sequence[Document]) -> None:
        postings: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
        self._doc_len = {}
        for doc in corpus:
            terms = content_terms(doc.full_text)
            self._doc_len[doc.doc_id] = len(terms)
            for term, tf in Counter(terms).items():
                postings[term].append((doc.doc_id, tf))
        self._postings = dict(postings)
        n_docs = len(self._doc_len)
        self._avg_len = sum(self._doc_len.values()) / n_docs if n_docs else 0.0
        self._idf = {
            term: math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self._postings.items()
        }

    def score(self, query: str) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for term, qtf in Counter(content_terms(query)).items():
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self._postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[doc_id] / self._avg_len)
                scores[doc_id] += qtf * idf * tf * (self.k1 + 1) / (tf + norm)
        return dict(scores)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        return top_k(self.score(query), k)

    def embedding_usage(self, query: str) -> dict[str, int]:
        return {}
