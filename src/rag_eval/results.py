"""Per-query result records and their per-pipeline summaries."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_eval.cost import cost_per_1k_queries
from rag_eval.metrics.latency import percentile

STAGES = ("retrieve", "rerank", "generate", "total")
RECORDS_FILE = "per_query.jsonl"
SUMMARY_FILE = "summary.csv"
MANIFEST_FILE = "manifest.json"

Record = dict[str, Any]


@dataclass
class PipelineSummary:
    name: str
    n_queries: int
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    cost_per_1k_usd: float
    mean_steps: float
    mean_input_tokens: float
    mean_output_tokens: float
    latency_modeled: bool
    quality_by_type: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {"pipeline": self.name, "n_queries": self.n_queries}
        row.update({k: round(v, 4) for k, v in self.metrics.items()})
        row.update({f"latency_{k}_ms": round(v, 2) for k, v in self.latency_ms.items()})
        row["cost_per_1k_usd"] = round(self.cost_per_1k_usd, 4)
        row["mean_steps"] = round(self.mean_steps, 3)
        row["mean_input_tokens"] = round(self.mean_input_tokens, 1)
        row["mean_output_tokens"] = round(self.mean_output_tokens, 1)
        row["latency_modeled"] = self.latency_modeled
        return row


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize(records: Sequence[Record], quality_metric: str) -> list[PipelineSummary]:
    """Aggregate records per pipeline, keeping pipelines in first-seen order."""
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record["pipeline"]].append(record)
    summaries = []
    for name, rows in grouped.items():
        metric_names = list(rows[0]["metrics"])
        metrics = {
            m: value
            for m in metric_names
            if (value := _mean(r["metrics"][m] for r in rows)) is not None
        }
        latency: dict[str, float] = {}
        for stage in STAGES:
            values = [r["latency_ms"][stage] for r in rows]
            latency[f"{stage}_p50"] = percentile(values, 50)
            latency[f"{stage}_p95"] = percentile(values, 95)
        by_type: dict[str, list[float | None]] = defaultdict(list)
        for r in rows:
            by_type[r["query_type"]].append(r["metrics"].get(quality_metric))
        summaries.append(
            PipelineSummary(
                name=name,
                n_queries=len(rows),
                metrics=metrics,
                latency_ms=latency,
                cost_per_1k_usd=cost_per_1k_queries(r["cost_usd"] for r in rows),
                mean_steps=sum(r["steps"] for r in rows) / len(rows),
                mean_input_tokens=sum(r["input_tokens"] for r in rows) / len(rows),
                mean_output_tokens=sum(r["output_tokens"] for r in rows) / len(rows),
                latency_modeled=any(r["latency_modeled"] for r in rows),
                quality_by_type={
                    t: v for t, vals in sorted(by_type.items()) if (v := _mean(vals)) is not None
                },
            )
        )
    return summaries


def paired_metric(
    records: Sequence[Record], pipeline_a: str, pipeline_b: str, metric: str
) -> tuple[list[float], list[float]]:
    """Per-query metric values for two pipelines, aligned on the queries both answered."""
    by_pipeline: dict[str, dict[str, float]] = defaultdict(dict)
    for r in records:
        value = r["metrics"].get(metric)
        if value is not None:
            by_pipeline[r["pipeline"]][r["query_id"]] = value
    for name in (pipeline_a, pipeline_b):
        if name not in by_pipeline:
            known = sorted({r["pipeline"] for r in records})
            raise KeyError(f"no {metric!r} values for pipeline {name!r}; pipelines: {known}")
    shared = sorted(set(by_pipeline[pipeline_a]) & set(by_pipeline[pipeline_b]))
    return (
        [by_pipeline[pipeline_a][q] for q in shared],
        [by_pipeline[pipeline_b][q] for q in shared],
    )


def write_records(path: Path, records: Iterable[Record]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(run_dir: Path) -> list[Record]:
    with (run_dir / RECORDS_FILE).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    return manifest


def write_summary_csv(path: Path, summaries: Sequence[PipelineSummary]) -> None:
    rows = [s.flat() for s in summaries]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
