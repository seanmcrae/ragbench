"""Paired comparison of two pipelines from one run, shared by the CLI and the docs site."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from rag_eval.report.render import Table
from rag_eval.results import Record, paired_metric, summarize
from rag_eval.stats import DeltaCI, paired_bootstrap

COMPARE_METRICS = ("answer_f1", "exact_match", "judge_score", "groundedness", "ndcg@5", "mrr")


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    ci: DeltaCI


@dataclass(frozen=True)
class Comparison:
    """Pipeline B measured against baseline A on the queries both answered."""

    a: str
    b: str
    gate_metric: str
    deltas: list[MetricDelta]
    cost_per_1k: tuple[float, float]
    p95_ms: tuple[float, float]

    @property
    def gate(self) -> DeltaCI | None:
        return next((d.ci for d in self.deltas if d.metric == self.gate_metric), None)

    @property
    def n_queries(self) -> int:
        gate = self.gate
        return gate.n if gate is not None else 0

    def is_regression(self, tolerance: float = 0.0) -> bool:
        gate = self.gate
        return gate is not None and gate.is_regression(tolerance)


def compare_pipelines(
    records: Sequence[Record],
    config: dict[str, Any],
    a: str,
    b: str,
    *,
    gate_metric: str | None = None,
    metrics: Sequence[str] = COMPARE_METRICS,
) -> Comparison:
    """Bootstrap every metric both pipelines report; the gate metric must be among them.

    Raises KeyError naming the known pipelines when A or B is not in the run.
    """
    gate = gate_metric or config["quality_metric"]
    deltas = []
    for metric in dict.fromkeys([gate, *metrics]):
        try:
            xs, ys = paired_metric(records, a, b, metric)
        except KeyError:
            if metric == gate:
                raise
            continue
        if xs:
            ci = paired_bootstrap(xs, ys, config["bootstrap_samples"], seed=config["seed"])
            deltas.append(MetricDelta(metric, ci))
    summaries = {s.name: s for s in summarize(records, gate)}
    sa, sb = summaries[a], summaries[b]
    return Comparison(
        a=a,
        b=b,
        gate_metric=gate,
        deltas=deltas,
        cost_per_1k=(sa.cost_per_1k_usd, sb.cost_per_1k_usd),
        p95_ms=(sa.latency_ms["total_p95"], sb.latency_ms["total_p95"]),
    )


def render_comparison_text(comparison: Comparison) -> str:
    c = comparison
    lines = [
        f"{c.b} vs {c.a} over {c.n_queries} queries (B - A, 95% CI)",
        f"{'metric':<14}{'A':>8}{'B':>8}{'delta':>9}",
    ]
    for d in c.deltas:
        ci = d.ci
        lines.append(
            f"{d.metric:<14}{ci.mean_a:>8.3f}{ci.mean_b:>8.3f}{ci.delta:>+9.3f}"
            f"   [{ci.low:+.3f}, {ci.high:+.3f}]   p={ci.p_value:.3f}"
        )
    (ca, cb), (pa, pb) = c.cost_per_1k, c.p95_ms
    lines.append(f"{'$/1k queries':<14}{ca:>8.3f}{cb:>8.3f}{cb - ca:>+9.3f}")
    lines.append(f"{'p95 ms':<14}{pa:>8.0f}{pb:>8.0f}{pb - pa:>+9.0f}")
    return "\n".join(lines)


def comparison_table(comparison: Comparison) -> Table:
    c = comparison
    rows = [
        [
            d.metric,
            f"{d.ci.mean_a:.3f}",
            f"{d.ci.mean_b:.3f}",
            f"{d.ci.delta:+.3f}",
            f"[{d.ci.low:+.3f}, {d.ci.high:+.3f}]",
            f"{d.ci.p_value:.3f}",
        ]
        for d in c.deltas
    ]
    (ca, cb), (pa, pb) = c.cost_per_1k, c.p95_ms
    rows.append(["$/1k queries", f"{ca:.3f}", f"{cb:.3f}", f"{cb - ca:+.3f}", "", ""])
    rows.append(["p95 ms", f"{pa:,.0f}", f"{pb:,.0f}", f"{pb - pa:+,.0f}", "", ""])
    return Table(["metric", c.a, c.b, "delta", "95% CI", "p"], rows)
