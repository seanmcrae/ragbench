"""Assemble everything a report shows from a persisted run, independent of output format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_eval.config import BudgetConfig
from rag_eval.pareto import Candidate, Recommendation, recommend
from rag_eval.results import PipelineSummary, Record, paired_metric, summarize
from rag_eval.stats import DeltaCI, paired_bootstrap


@dataclass(frozen=True)
class BaselineDelta:
    pipeline: str
    ci: DeltaCI


@dataclass(frozen=True)
class ReportData:
    experiment: str
    manifest: dict[str, Any]
    quality_metric: str
    baseline: str
    summaries: list[PipelineSummary]
    candidates: list[Candidate]
    recommendation: Recommendation
    deltas: list[BaselineDelta]

    @property
    def latency_modeled(self) -> bool:
        return any(s.latency_modeled for s in self.summaries)

    @property
    def query_types(self) -> list[str]:
        return sorted({t for s in self.summaries for t in s.quality_by_type})


def build_report(
    records: list[Record],
    manifest: dict[str, Any],
    *,
    quality_metric: str | None = None,
    budget: BudgetConfig | None = None,
    baseline: str | None = None,
) -> ReportData:
    config = manifest["config"]
    metric = quality_metric or config["quality_metric"]
    summaries = summarize(records, metric)
    if any(metric not in s.metrics for s in summaries):
        available = sorted(summaries[0].metrics) if summaries else []
        raise KeyError(f"metric {metric!r} not in results; available: {available}")
    budget = budget or BudgetConfig.model_validate(config["budget"])
    baseline = baseline or config.get("baseline") or summaries[0].name
    candidates = [
        Candidate(s.name, s.metrics[metric], s.cost_per_1k_usd, s.latency_ms["total_p95"])
        for s in summaries
    ]
    deltas = []
    for s in summaries:
        if s.name == baseline:
            continue
        a, b = paired_metric(records, baseline, s.name, metric)
        ci = paired_bootstrap(a, b, config["bootstrap_samples"], seed=config["seed"])
        deltas.append(BaselineDelta(s.name, ci))
    return ReportData(
        experiment=manifest["experiment"],
        manifest=manifest,
        quality_metric=metric,
        baseline=baseline,
        summaries=summaries,
        candidates=candidates,
        recommendation=recommend(candidates, budget),
        deltas=deltas,
    )
