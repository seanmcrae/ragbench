"""Retriever contract shared by lexical, dense and hybrid implementations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval.datasets import Document


@dataclass(frozen=True, slots=True)
class ScoredDoc:
    doc_id: str
    score: float


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def index(self, corpus: Sequence[Document]) -> None:
        """Build whatever index the retriever needs; called once per corpus."""

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Top-``k`` documents, best first."""

    def embedding_usage(self, query: str) -> dict[str, int]:
        """Billable embedding tokens for one query, keyed by priced model name."""


def top_k(scores: dict[str, float], k: int) -> list[ScoredDoc]:
    """Deterministic top-k: higher score first, ties broken by doc id."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [ScoredDoc(doc_id, score) for doc_id, score in ranked[:k]]
