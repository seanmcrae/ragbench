"""Hybrid retrieval by reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rag_eval.datasets import Document
from rag_eval.retrieval.base import Retriever, ScoredDoc, top_k


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredDoc]:
    """Fuse ranked lists with score(d) = sum_i w_i / (k + rank_i(d)), ranks starting at 1.

    RRF ignores raw scores, which is the point: BM25 and cosine scores live on incomparable
    scales, while ranks do not.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += weight / (k + rank)
    return top_k(dict(fused), len(fused))


class HybridRetriever:
    def __init__(
        self,
        components: Sequence[Retriever],
        rrf_k: int = 60,
        weights: Sequence[float] | None = None,
        candidate_depth: int = 50,
    ) -> None:
        if len(components) < 2:
            raise ValueError("hybrid retrieval needs at least two components")
        self.components = list(components)
        self.rrf_k = rrf_k
        self.weights = list(weights) if weights is not None else None
        self.candidate_depth = candidate_depth

    @property
    def name(self) -> str:
        return "hybrid[" + "+".join(c.name for c in self.components) + "]"

    def index(self, corpus: Sequence[Document]) -> None:
        for component in self.components:
            component.index(corpus)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        depth = max(k, self.candidate_depth)
        rankings = [[hit.doc_id for hit in c.search(query, depth)] for c in self.components]
        return reciprocal_rank_fusion(rankings, self.rrf_k, self.weights)[:k]

    def embedding_usage(self, query: str) -> dict[str, int]:
        usage: defaultdict[str, int] = defaultdict(int)
        for component in self.components:
            for model, tokens in component.embedding_usage(query).items():
                usage[model] += tokens
        return dict(usage)
