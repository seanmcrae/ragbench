from collections.abc import Sequence
from pathlib import Path

from rag_eval.cache import CachedGenerator, CachedJudge, ResponseCache, cache_key
from rag_eval.datasets import Dataset, Document, Query
from rag_eval.datasets.synthetic import build_synthetic_dataset
from rag_eval.generation import Completion, ExtractiveGenerator
from rag_eval.judge import HeuristicJudge
from rag_eval.pipeline import Pipeline
from rag_eval.retrieval import BM25Retriever
from rag_eval.retrieval.rerank import QueryLikelihoodReranker


class CountingGenerator(ExtractiveGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def answer(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str] = ()
    ) -> Completion:
        self.calls += 1
        return super().answer(question, contexts, searches)


def test_cache_key_is_order_sensitive_and_stable() -> None:
    assert cache_key("a", [1, 2]) == cache_key("a", [1, 2])
    assert cache_key("a", [1, 2]) != cache_key("a", [2, 1])


def test_cached_generator_serves_repeat_calls(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    inner = CountingGenerator()
    gen = CachedGenerator(inner, cache)
    docs = [Document("d", "", "The sync interval is every 5 minutes.")]
    first = gen.answer("What is the sync interval?", docs)
    second = gen.answer("What is the sync interval?", docs)
    assert first == second
    assert inner.calls == 1
    assert (cache.hits, cache.misses) == (1, 1)
    # Editing a passage's text changes the key even though the doc id is the same.
    gen.answer("What is the sync interval?", [Document("d", "", "The interval is hourly.")])
    assert inner.calls == 2
    cache.close()
    # The cache persists across processes.
    reopened = CachedGenerator(CountingGenerator(), ResponseCache(tmp_path / "c.sqlite"))
    assert reopened.answer("What is the sync interval?", docs) == first


def test_cached_judge(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.sqlite")
    judge = CachedJudge(HeuristicJudge(), cache)
    docs = [Document("d", "", "x")]
    assert judge.judge("q", "a", ["a"], docs) == judge.judge("q", "a", ["a"], docs)
    assert cache.hits == 1
    assert judge.name == "heuristic-rubric-v1"


def test_single_step_pipeline_trace(tiny_dataset: Dataset) -> None:
    pipeline = Pipeline("bm25", BM25Retriever(), ExtractiveGenerator(), top_k=2)
    pipeline.index(list(tiny_dataset.corpus.values()))
    trace = pipeline.run(tiny_dataset.queries["q1"])
    assert trace.contexts == ["d1", "d2"]
    assert trace.answer == "300 requests [d1]"
    assert trace.steps == 1
    assert len(trace.usage) == 1
    assert trace.latency_modeled
    assert trace.total_ms >= trace.stage_ms["generate"] > 0


def test_reranker_stage_is_timed(tiny_dataset: Dataset) -> None:
    pipeline = Pipeline(
        "rr", BM25Retriever(), ExtractiveGenerator(), top_k=2, reranker=QueryLikelihoodReranker()
    )
    pipeline.index(list(tiny_dataset.corpus.values()))
    trace = pipeline.run(tiny_dataset.queries["q1"])
    assert trace.stage_ms["rerank"] > 0
    assert trace.contexts[0] == "d1"


def test_agentic_pipeline_resolves_multi_hop_query() -> None:
    ds = build_synthetic_dataset()
    question = Query(
        "mh",
        "What is the audit log retention on the cheapest plan that includes custom roles?",
    )
    single = Pipeline("single", BM25Retriever(), ExtractiveGenerator(), top_k=3)
    agent = Pipeline("agent", BM25Retriever(), ExtractiveGenerator(), top_k=3, max_steps=3)
    for p in (single, agent):
        p.index(list(ds.corpus.values()))
    agent_trace = agent.run(question)
    assert agent_trace.searches[0] == "cheapest include custom role"
    assert agent_trace.steps == 3
    assert "plan-business-audit-retention" in agent_trace.contexts
    assert agent_trace.answer == "1 year [plan-business-audit-retention]"
    assert len(agent_trace.usage) == 3  # two planner calls (budget used up) plus the answer
    assert single.run(question).answer != agent_trace.answer
