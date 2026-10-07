"""Command-line interface: ``rag-eval run``, ``rag-eval report``, ``rag-eval compare``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rag_eval.config import BudgetConfig, load_config
from rag_eval.report import build_report, write_report
from rag_eval.report.model import ReportData
from rag_eval.report.render import recommendation_paragraphs, render_text_table, summary_table
from rag_eval.results import load_manifest, load_records, paired_metric, summarize
from rag_eval.runner import resolve_run_dir, run_experiment
from rag_eval.stats import paired_bootstrap

app = typer.Typer(
    help="Compare RAG and agentic search pipelines on quality, latency and cost.",
    no_args_is_help=True,
    add_completion=False,
)

RunsDir = Annotated[Path, typer.Option("--runs-dir", help="Where runs are stored.")]
RunOpt = Annotated[
    Path | None, typer.Option("--run", help="Run directory (default: the latest run).")
]

COMPARE_METRICS = ("answer_f1", "exact_match", "judge_score", "groundedness", "ndcg@5", "mrr")


def _plain(text: str) -> str:
    return text.replace("**", "")


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Experiment YAML.")],
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress output.")] = False,
) -> None:
    """Run every pipeline in CONFIG over its dataset and persist per-query results."""
    experiment = load_config(config)

    def progress(pipeline: str, done: int, total: int) -> None:
        if done == total:
            typer.echo(f"  {pipeline}: {total} queries", err=True)

    run_dir = run_experiment(experiment, progress=None if quiet else progress)
    records = load_records(run_dir)
    manifest = load_manifest(run_dir)
    data = build_report(records, manifest)
    typer.echo(render_text_table(summary_table(data)))
    cache = manifest["cache"]
    typer.echo(f"\nresults: {run_dir}  (cache hits {cache['hits']}, misses {cache['misses']})")


@app.command()
def report(
    run_dir: Annotated[Path | None, typer.Argument(help="Run directory (default: latest).")] = None,
    runs_dir: RunsDir = Path("runs"),
    img_dir: Annotated[
        Path | None, typer.Option(help="Also copy the chart here, e.g. docs/img.")
    ] = None,
    metric: Annotated[str | None, typer.Option(help="Quality metric to rank by.")] = None,
    max_cost: Annotated[
        float | None, typer.Option(help="Override budget: USD per 1k queries.")
    ] = None,
    max_p95: Annotated[float | None, typer.Option(help="Override budget: p95 ms.")] = None,
) -> None:
    """Write report.md, report.html and the quality-vs-cost chart for a run."""
    resolved = resolve_run_dir(run_dir, runs_dir)
    manifest = load_manifest(resolved)
    budget = None
    if max_cost is not None or max_p95 is not None:
        configured = BudgetConfig.model_validate(manifest["config"]["budget"])
        budget = BudgetConfig(
            max_cost_per_1k_usd=max_cost
            if max_cost is not None
            else configured.max_cost_per_1k_usd,
            max_p95_latency_ms=max_p95 if max_p95 is not None else configured.max_p95_latency_ms,
        )
    data: ReportData = build_report(
        load_records(resolved), manifest, quality_metric=metric, budget=budget
    )
    written = write_report(data, resolved, img_dir)
    for paragraph in recommendation_paragraphs(data):
        typer.echo(_plain(paragraph))
    typer.echo(f"\nwrote {written.markdown}, {written.html}, {written.chart}")


@app.command()
def compare(
    a: Annotated[str, typer.Argument(help="Baseline pipeline name.")],
    b: Annotated[str, typer.Argument(help="Candidate pipeline name.")],
    run: RunOpt = None,
    runs_dir: RunsDir = Path("runs"),
    gate_metric: Annotated[
        str | None, typer.Option("--metric", help="Metric the release gate checks.")
    ] = None,
    tolerance: Annotated[
        float, typer.Option(help="Largest acceptable drop in the gate metric.")
    ] = 0.0,
    fail_on_regression: Annotated[
        bool, typer.Option(help="Exit 1 if B is significantly worse than A on the gate metric.")
    ] = False,
) -> None:
    """Paired comparison of pipeline B against A, with bootstrap confidence intervals."""
    resolved = resolve_run_dir(run, runs_dir)
    records = load_records(resolved)
    config = load_manifest(resolved)["config"]
    gate = gate_metric or config["quality_metric"]
    metrics = list(dict.fromkeys([gate, *COMPARE_METRICS]))
    rows = []
    gate_ci = None
    for metric in metrics:
        try:
            xs, ys = paired_metric(records, a, b, metric)
        except KeyError:
            if metric == gate:
                raise
            continue
        if not xs:
            continue
        ci = paired_bootstrap(xs, ys, config["bootstrap_samples"], seed=config["seed"])
        if metric == gate:
            gate_ci = ci
        rows.append(
            f"{metric:<14}{ci.mean_a:>8.3f}{ci.mean_b:>8.3f}{ci.delta:>+9.3f}"
            f"   [{ci.low:+.3f}, {ci.high:+.3f}]   p={ci.p_value:.3f}"
        )
    summaries = {s.name: s for s in summarize(records, gate)}
    sa, sb = summaries[a], summaries[b]
    typer.echo(f"{b} vs {a} over {gate_ci.n if gate_ci else 0} queries (B - A, 95% CI)")
    typer.echo(f"{'metric':<14}{'A':>8}{'B':>8}{'delta':>9}")
    typer.echo("\n".join(rows))
    typer.echo(
        f"{'$/1k queries':<14}{sa.cost_per_1k_usd:>8.3f}{sb.cost_per_1k_usd:>8.3f}"
        f"{sb.cost_per_1k_usd - sa.cost_per_1k_usd:>+9.3f}"
    )
    pa, pb = sa.latency_ms["total_p95"], sb.latency_ms["total_p95"]
    typer.echo(f"{'p95 ms':<14}{pa:>8.0f}{pb:>8.0f}{pb - pa:>+9.0f}")
    if gate_ci is not None and gate_ci.is_regression(tolerance):
        typer.echo(f"\nGATE FAIL: {b} is significantly worse than {a} on {gate}.")
        if fail_on_regression:
            raise typer.Exit(code=1)
    else:
        typer.echo(f"\nGATE PASS: no significant drop larger than {tolerance} on {gate}.")


if __name__ == "__main__":  # pragma: no cover
    app()
