"""Run every configured pipeline over the dataset, score each query, persist the results."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from rag_eval import __version__
from rag_eval.cache import CachedGenerator, CachedJudge, ResponseCache
from rag_eval.config import ExperimentConfig
from rag_eval.cost import PriceTable
from rag_eval.datasets import Dataset, Document, Query
from rag_eval.factory import build_generator, build_judge, build_pipeline, load_dataset
from rag_eval.judge import Judge
from rag_eval.metrics import exact_match, groundedness, retrieval_metrics, token_f1
from rag_eval.metrics.groundedness import CITATION
from rag_eval.pipeline import QueryTrace
from rag_eval.results import (
    MANIFEST_FILE,
    RECORDS_FILE,
    SUMMARY_FILE,
    Record,
    summarize,
    write_records,
    write_summary_csv,
)

LATEST_FILE = "LATEST"
Progress = Callable[[str, int, int], None]


def score_trace(
    pipeline: str,
    query: Query,
    trace: QueryTrace,
    dataset: Dataset,
    judge: Judge,
    prices: PriceTable,
    ks: list[int],
    docs: Mapping[str, Document],
) -> Record:
    metrics: dict[str, float | None] = dict(
        retrieval_metrics(trace.ranking, dataset.qrels[query.query_id], ks)
    )
    references = dataset.answers.get(query.query_id, [])
    answer_text = " ".join(CITATION.sub(" ", trace.answer).split())
    metrics["answer_f1"] = token_f1(answer_text, references) if references else None
    metrics["exact_match"] = exact_match(answer_text, references) if references else None
    grounded = groundedness(trace.answer, docs, set(trace.contexts))
    metrics["groundedness"] = grounded.score
    metrics["citation_rate"] = grounded.citation_rate
    contexts = [docs[d] for d in trace.contexts]
    verdict = judge.judge(query.text, trace.answer, references, contexts)
    metrics["judge_score"] = verdict.normalized
    stage_ms = dict(trace.stage_ms)
    stage_ms["total"] = trace.total_ms
    return {
        "pipeline": pipeline,
        "query_id": query.query_id,
        "query_type": query.metadata.get("type", "all"),
        "question": query.text,
        "answer": trace.answer,
        "references": references,
        "contexts": trace.contexts,
        "ranking": trace.ranking[: max(ks)],
        "searches": trace.searches,
        "steps": trace.steps,
        "metrics": metrics,
        "judge_rationale": verdict.rationale,
        "latency_ms": stage_ms,
        "latency_modeled": trace.latency_modeled,
        "input_tokens": sum(u.input_tokens for u in trace.usage),
        "output_tokens": sum(u.output_tokens for u in trace.usage),
        "embedding_tokens": dict(trace.embedding_tokens),
        "cost_usd": prices.query_cost(trace.usage, trace.embedding_tokens),
    }


def run_experiment(config: ExperimentConfig, progress: Progress | None = None) -> Path:
    dataset = load_dataset(config.dataset)
    prices = PriceTable.load(config.prices)
    cache = ResponseCache(config.cache_dir / "responses.sqlite")
    judge = CachedJudge(build_judge(config.judge), cache)
    corpus = list(dataset.corpus.values())
    queries = dataset.evaluable_queries()
    records: list[Record] = []
    try:
        for pipeline_cfg in config.pipelines:
            generator = CachedGenerator(build_generator(pipeline_cfg.generator), cache)
            pipeline = build_pipeline(
                pipeline_cfg, generator, config.seed, retrieval_depth=max(config.metrics_k)
            )
            pipeline.index(corpus)
            for i, query in enumerate(queries, start=1):
                trace = pipeline.run(query)
                records.append(
                    score_trace(
                        pipeline.name,
                        query,
                        trace,
                        dataset,
                        judge,
                        prices,
                        config.metrics_k,
                        dataset.corpus,
                    )
                )
                if progress is not None:
                    progress(pipeline.name, i, len(queries))
        cache_stats = {"hits": cache.hits, "misses": cache.misses}
    finally:
        cache.close()

    run_dir = config.output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_records(run_dir / RECORDS_FILE, records)
    write_summary_csv(run_dir / SUMMARY_FILE, summarize(records, config.quality_metric))
    manifest = {
        "experiment": config.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rag_eval_version": __version__,
        "dataset": {
            "name": dataset.name,
            "fingerprint": dataset.fingerprint(),
            "documents": len(corpus),
            "queries": len(queries),
            "has_reference_answers": dataset.has_answers,
        },
        "prices_as_of": prices.as_of,
        "cache": cache_stats,
        "config": config.model_dump(mode="json"),
    }
    (run_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (config.output_dir / LATEST_FILE).write_text(config.name + "\n", encoding="utf-8")
    return run_dir


def resolve_run_dir(run: Path | None, output_dir: Path = Path("runs")) -> Path:
    """An explicit run directory, or the most recent run recorded under ``output_dir``."""
    if run is not None:
        return run
    latest = output_dir / LATEST_FILE
    if not latest.exists():
        raise FileNotFoundError(f"no run found: {latest} is missing; run an experiment first")
    return output_dir / latest.read_text(encoding="utf-8").strip()
