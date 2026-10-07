"""Generator contract: answer from retrieved passages, or propose the next search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_eval.cost import TokenUsage
from rag_eval.datasets import Document


@dataclass(frozen=True)
class Completion:
    """Output of one generator step.

    ``latency_ms`` is wall-clock for API providers. The offline mock reports a *modeled*
    latency instead (``latency_modeled=True``) so latency trade-offs stay visible without keys;
    reports label modeled numbers as such.
    """

    text: str
    usage: tuple[TokenUsage, ...] = field(default_factory=tuple)
    latency_ms: float = 0.0
    latency_modeled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": [[u.model, u.input_tokens, u.output_tokens] for u in self.usage],
            "latency_ms": self.latency_ms,
            "latency_modeled": self.latency_modeled,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Completion:
        return cls(
            text=raw["text"],
            usage=tuple(TokenUsage(m, int(i), int(o)) for m, i, o in raw["usage"]),
            latency_ms=float(raw["latency_ms"]),
            latency_modeled=bool(raw["latency_modeled"]),
        )


class Generator(Protocol):
    @property
    def name(self) -> str:
        """Stable identity including model and parameters; part of every cache key."""

    def answer(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str] = ()
    ) -> Completion:
        """Answer ``question`` from ``contexts``, citing passages as ``[doc-id]``.

        ``searches`` lists follow-up queries already issued in agentic mode.
        """

    def next_search(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str]
    ) -> Completion:
        """Propose one follow-up search query, or return empty text when ready to answer."""
