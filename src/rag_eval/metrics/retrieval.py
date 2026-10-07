"""Ranked-retrieval metrics over graded relevance judgements.

Documents with grade > 0 count as relevant for binary metrics (recall, precision, MRR);
nDCG uses the grades directly with exponential gain 2^g - 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _relevant(qrels: Mapping[str, int]) -> set[str]:
    return {doc_id for doc_id, grade in qrels.items() if grade > 0}


def recall_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    relevant = _relevant(qrels)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Denominator is k even when fewer than k documents were returned (trec_eval semantics)."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(_relevant(qrels).intersection(ranking[:k])) / k


def reciprocal_rank(
    ranking: Sequence[str], qrels: Mapping[str, int], k: int | None = None
) -> float:
    relevant = _relevant(qrels)
    for rank, doc_id in enumerate(ranking[:k] if k else ranking, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = [2 ** qrels.get(doc_id, 0) - 1 for doc_id in ranking[:k]]
    ideal = sorted((2**grade - 1 for grade in qrels.values() if grade > 0), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0


def retrieval_metrics(
    ranking: Sequence[str], qrels: Mapping[str, int], ks: Sequence[int]
) -> dict[str, float]:
    out: dict[str, float] = {"mrr": reciprocal_rank(ranking, qrels)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranking, qrels, k)
        out[f"precision@{k}"] = precision_at_k(ranking, qrels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranking, qrels, k)
    return out
