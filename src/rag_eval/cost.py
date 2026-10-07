"""Token usage accounting and cost-to-serve from a configurable price table."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

TOKENS_PER_UNIT = 1_000_000


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """One billable model call."""

    model: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class PriceTable:
    as_of: str
    currency: str
    models: Mapping[str, ModelPrice]

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw.get("unit", "per_million_tokens") != "per_million_tokens":
            raise ValueError("price table unit must be per_million_tokens")
        models = {
            name: ModelPrice(float(p["input"]), float(p.get("output", 0.0)))
            for name, p in raw["models"].items()
        }
        return cls(as_of=str(raw["as_of"]), currency=str(raw.get("currency", "USD")), models=models)

    def price(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            known = ", ".join(sorted(self.models))
            raise ValueError(f"no price for model {model!r}; known models: {known}") from None

    def call_cost(self, usage: TokenUsage) -> float:
        price = self.price(usage.model)
        return (
            usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
        ) / TOKENS_PER_UNIT

    def query_cost(
        self, calls: Iterable[TokenUsage], embedding_tokens: Mapping[str, int] | None = None
    ) -> float:
        """Serving cost of one query: every LLM call plus query-time embedding tokens.

        Corpus indexing is a one-off cost and is deliberately excluded from cost-to-serve.
        """
        total = sum(self.call_cost(call) for call in calls)
        for model, tokens in (embedding_tokens or {}).items():
            total += tokens * self.price(model).input_per_mtok / TOKENS_PER_UNIT
        return total


def cost_per_1k_queries(per_query_costs: Iterable[float]) -> float:
    costs = list(per_query_costs)
    return 1000 * sum(costs) / len(costs) if costs else 0.0
