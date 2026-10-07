"""Pareto frontier over (quality up, cost down, latency down) and budgeted selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_eval.config import BudgetConfig


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    quality: float
    cost_per_1k: float
    p95_ms: float

    def dominates(self, other: Candidate) -> bool:
        no_worse = (
            self.quality >= other.quality
            and self.cost_per_1k <= other.cost_per_1k
            and self.p95_ms <= other.p95_ms
        )
        strictly_better = (
            self.quality > other.quality
            or self.cost_per_1k < other.cost_per_1k
            or self.p95_ms < other.p95_ms
        )
        return no_worse and strictly_better


def pareto_frontier(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Non-dominated candidates, cheapest first."""
    frontier = [c for c in candidates if not any(o.dominates(c) for o in candidates)]
    return sorted(frontier, key=lambda c: (c.cost_per_1k, c.p95_ms, -c.quality, c.name))


@dataclass
class Recommendation:
    budget: BudgetConfig
    chosen: Candidate | None
    frontier: list[Candidate]
    rejected: dict[str, str] = field(default_factory=dict)


def recommend(candidates: Sequence[Candidate], budget: BudgetConfig) -> Recommendation:
    """Highest-quality candidate inside the budget; ties go to cheaper, then faster.

    The chosen config is always on the frontier of the in-budget set. Out-of-budget configs
    are listed with the constraint they break so the report can show what the budget costs.
    """
    rejected: dict[str, str] = {}
    eligible = []
    for c in candidates:
        reasons = []
        if budget.max_cost_per_1k_usd is not None and c.cost_per_1k > budget.max_cost_per_1k_usd:
            reasons.append(f"cost ${c.cost_per_1k:.3f}/1k > ${budget.max_cost_per_1k_usd:.2f}")
        if budget.max_p95_latency_ms is not None and c.p95_ms > budget.max_p95_latency_ms:
            reasons.append(f"p95 {c.p95_ms:.0f} ms > {budget.max_p95_latency_ms:.0f} ms")
        if reasons:
            rejected[c.name] = "; ".join(reasons)
        else:
            eligible.append(c)
    frontier = pareto_frontier(eligible)
    chosen = (
        min(frontier, key=lambda c: (-c.quality, c.cost_per_1k, c.p95_ms, c.name))
        if frontier
        else None
    )
    return Recommendation(budget, chosen, frontier, rejected)
