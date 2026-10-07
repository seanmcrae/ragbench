import json
from pathlib import Path
from typing import Any

import pytest

from rag_eval.config import parse_config
from rag_eval.datasets import Dataset, save_beir_dir
from rag_eval.results import load_manifest, load_records, paired_metric, summarize
from rag_eval.runner import resolve_run_dir, run_experiment

ROOT = Path(__file__).resolve().parent.parent


def experiment(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "smoke",
        "dataset": {"kind": "synthetic", "limit": 12},
        "prices": str(ROOT / "configs" / "prices.yaml"),
        "output_dir": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"),
        "metrics_k": [1, 5],
        "pipelines": [
            {"name": "bm25-k3", "retriever": {"kind": "bm25"}, "top_k": 3},
            {"name": "agent", "retriever": {"kind": "bm25"}, "top_k": 3, "max_steps": 2},
        ],
    }
    raw.update(overrides)
    return raw


def test_run_persists_records_summary_and_manifest(tmp_path: Path) -> None:
    run_dir = run_experiment(parse_config(experiment(tmp_path)))
    assert run_dir == tmp_path / "runs" / "smoke"
    assert resolve_run_dir(None, tmp_path / "runs") == run_dir
    records = load_records(run_dir)
    assert len(records) == 24
    first = records[0]
    assert set(first["metrics"]) >= {"ndcg@5", "recall@1", "mrr", "answer_f1", "judge_score"}
    assert first["cost_usd"] > 0
    assert first["latency_ms"]["total"] >= first["latency_ms"]["generate"]
    assert (run_dir / "summary.csv").read_text().startswith("pipeline,n_queries,mrr,")
    manifest = load_manifest(run_dir)
    assert manifest["dataset"]["queries"] == 12
    assert manifest["prices_as_of"] == "2026-10-07"
    assert manifest["cache"]["misses"] > 0


def test_rerun_is_served_from_cache_with_identical_results(tmp_path: Path) -> None:
    cfg = parse_config(experiment(tmp_path))
    first = load_records(run_experiment(cfg))
    second_dir = run_experiment(cfg)
    second = load_records(second_dir)
    assert load_manifest(second_dir)["cache"]["misses"] == 0

    def stable(r: dict[str, Any]) -> tuple[Any, ...]:
        return (r["answer"], json.dumps(r["metrics"]), r["cost_usd"], r["latency_ms"]["generate"])

    assert [stable(r) for r in first] == [stable(r) for r in second]


def test_summary_and_pairing(tmp_path: Path) -> None:
    records = load_records(run_experiment(parse_config(experiment(tmp_path))))
    summaries = {s.name: s for s in summarize(records, "answer_f1")}
    assert set(summaries) == {"bm25-k3", "agent"}
    assert summaries["agent"].mean_steps >= 1
    assert summaries["agent"].cost_per_1k_usd >= summaries["bm25-k3"].cost_per_1k_usd
    assert summaries["bm25-k3"].latency_modeled
    a, b = paired_metric(records, "bm25-k3", "agent", "answer_f1")
    assert len(a) == len(b) == 12
    with pytest.raises(KeyError, match="pipelines"):
        paired_metric(records, "bm25-k3", "nope", "answer_f1")


def test_resolve_run_dir_without_runs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_run_dir(None, tmp_path)
    assert resolve_run_dir(tmp_path / "x") == tmp_path / "x"


def test_dataset_without_reference_answers(tmp_path: Path, tiny_dataset: Dataset) -> None:
    """BEIR-style data (e.g. SciFact) has qrels but no answers: answer metrics are skipped."""
    no_answers = Dataset("beir-like", tiny_dataset.corpus, tiny_dataset.queries, tiny_dataset.qrels)
    save_beir_dir(no_answers, tmp_path / "beir")
    raw = experiment(
        tmp_path,
        dataset={"kind": "beir", "path": str(tmp_path / "beir")},
        quality_metric="ndcg@5",
    )
    run_dir = run_experiment(parse_config(raw))
    records = load_records(run_dir)
    assert all(r["metrics"]["answer_f1"] is None for r in records)
    assert all(r["metrics"]["judge_score"] <= 0.5 for r in records)  # rubric caps at 3 of 5
    summary = summarize(records, "ndcg@5")[0]
    assert "answer_f1" not in summary.metrics
    assert not load_manifest(run_dir)["dataset"]["has_reference_answers"]
