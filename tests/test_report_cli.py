import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.config import BudgetConfig
from rag_eval.report import build_report, write_report
from rag_eval.report.render import Table, render_text_table
from rag_eval.results import load_manifest, load_records

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One small experiment run through the CLI, shared by the tests below."""
    root = tmp_path_factory.mktemp("cli")
    config = {
        "name": "cli-smoke",
        "dataset": {"kind": "synthetic", "limit": 15},
        "prices": str(ROOT / "configs" / "prices.yaml"),
        "output_dir": str(root / "runs"),
        "cache_dir": str(root / "cache"),
        "baseline": "bm25-k3",
        "budget": {"max_cost_per_1k_usd": 1.0, "max_p95_latency_ms": 1500},
        "bootstrap_samples": 500,
        "defaults": {"top_k": 3},
        "pipelines": [
            {"name": "bm25-k3", "retriever": {"kind": "bm25"}},
            {"name": "dense-k3", "retriever": {"kind": "dense"}},
            {"name": "agent-k3", "retriever": {"kind": "bm25"}, "max_steps": 3},
        ],
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(config))
    result = runner.invoke(app, ["run", str(path), "--quiet"])
    assert result.exit_code == 0, result.output
    assert "bm25-k3" in result.output and "results:" in result.output
    return root


def test_report_command_writes_all_artifacts(workdir: Path) -> None:
    img_dir = workdir / "img"
    result = runner.invoke(
        app, ["report", "--runs-dir", str(workdir / "runs"), "--img-dir", str(img_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "Recommended:" in result.output
    run_dir = workdir / "runs" / "cli-smoke"
    markdown = (run_dir / "report.md").read_text()
    assert markdown.startswith("# Evaluation report: cli-smoke")
    assert "## Versus baseline (bm25-k3)" in markdown
    assert "![Quality vs cost](quality_vs_cost.png)" in markdown
    page = (run_dir / "report.html").read_text()
    assert "data:image/png;base64," in page and "<strong>" in page
    assert (img_dir / "quality_vs_cost.png").read_bytes()[:4] == b"\x89PNG"


def test_report_budget_override_changes_recommendation(workdir: Path) -> None:
    run_dir = workdir / "runs" / "cli-smoke"
    records, manifest = load_records(run_dir), load_manifest(run_dir)
    tight = build_report(records, manifest, budget=BudgetConfig(max_cost_per_1k_usd=0.0001))
    assert tight.recommendation.chosen is None
    loose = build_report(records, manifest, budget=BudgetConfig())
    assert loose.recommendation.chosen is not None
    best = max(c.quality for c in loose.candidates)
    assert loose.recommendation.chosen.quality == best
    assert [d.pipeline for d in loose.deltas] == ["dense-k3", "agent-k3"]
    written = write_report(tight, run_dir)
    assert "No configuration fits the budget" in written.markdown.read_text()
    if loose.recommendation.chosen.name != loose.baseline:
        assert "Versus the baseline bm25-k3" in write_report(loose, run_dir).markdown.read_text()


def test_report_rejects_unknown_metric(workdir: Path) -> None:
    run_dir = workdir / "runs" / "cli-smoke"
    with pytest.raises(KeyError, match="available"):
        build_report(load_records(run_dir), load_manifest(run_dir), quality_metric="bleu")


def test_compare_prints_intervals_and_gate(workdir: Path) -> None:
    args = ["compare", "bm25-k3", "agent-k3", "--runs-dir", str(workdir / "runs")]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "agent-k3 vs bm25-k3 over 15 queries" in result.output
    assert "answer_f1" in result.output and "$/1k queries" in result.output
    assert "GATE" in result.output


def _write_fake_run(run_dir: Path, a_scores: list[float], b_scores: list[float]) -> None:
    run_dir.mkdir(parents=True)
    records = [
        {
            "pipeline": name,
            "query_id": f"q{i:02d}",
            "query_type": "all",
            "metrics": {"answer_f1": score, "mrr": 1.0},
            "latency_ms": {"retrieve": 1.0, "rerank": 0.0, "generate": 10.0, "total": 11.0},
            "cost_usd": 0.001,
            "steps": 1,
            "input_tokens": 100,
            "output_tokens": 5,
            "latency_modeled": True,
        }
        for name, scores in (("a", a_scores), ("b", b_scores))
        for i, score in enumerate(scores)
    ]
    (run_dir / "per_query.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    manifest = {"config": {"quality_metric": "answer_f1", "bootstrap_samples": 500, "seed": 0}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


def test_compare_gate_fails_on_significant_regression(tmp_path: Path) -> None:
    run_dir = tmp_path / "fake"
    _write_fake_run(run_dir, [1.0] * 30, [1.0] * 20 + [0.0] * 10)
    args = ["compare", "a", "b", "--run", str(run_dir)]
    warn_only = runner.invoke(app, args)
    assert warn_only.exit_code == 0
    assert "GATE FAIL" in warn_only.output
    assert runner.invoke(app, [*args, "--fail-on-regression"]).exit_code == 1
    # A drop smaller than the tolerance passes the gate even though it is significant.
    tolerant = runner.invoke(app, [*args, "--fail-on-regression", "--tolerance", "0.5"])
    assert tolerant.exit_code == 0
    assert "GATE PASS" in tolerant.output


def test_compare_identical_pipelines_pass(workdir: Path) -> None:
    args = ["compare", "agent-k3", "agent-k3", "--runs-dir", str(workdir / "runs")]
    result = runner.invoke(app, [*args, "--fail-on-regression"])
    assert result.exit_code == 0
    assert "GATE PASS" in result.output


def test_compare_unknown_pipeline_errors(workdir: Path) -> None:
    result = runner.invoke(app, ["compare", "bm25-k3", "nope", "--runs-dir", str(workdir / "runs")])
    assert result.exit_code != 0


def test_text_table_alignment() -> None:
    table = Table(["name", "value"], [["a", "1.0"], ["longer", "10.25"]])
    assert render_text_table(table).splitlines() == [
        "name    value",
        "a         1.0",
        "longer  10.25",
    ]
