"""A configured retrieve -> (rerank) -> generate pipeline, optionally agentic."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rag_eval.cost import TokenUsage
from rag_eval.datasets import Document, Query
from rag_eval.generation.base import Completion, Generator
from rag_eval.retrieval.base import Retriever
from rag_eval.retrieval.rerank import Reranker


@dataclass
class QueryTrace:
    """Everything one pipeline did for one query; scoring happens downstream."""

    query_id: str
    ranking: list[str]
    contexts: list[str]
    answer: str
    searches: list[str] = field(default_factory=list)
    usage: list[TokenUsage] = field(default_factory=list)
    embedding_tokens: Counter[str] = field(default_factory=Counter)
    stage_ms: dict[str, float] = field(
        default_factory=lambda: {"retrieve": 0.0, "rerank": 0.0, "generate": 0.0}
    )
    latency_modeled: bool = False

    @property
    def steps(self) -> int:
        return 1 + len(self.searches)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


class Pipeline:
    def __init__(
        self,
        name: str,
        retriever: Retriever,
        generator: Generator,
        top_k: int = 5,
        reranker: Reranker | None = None,
        rerank_depth: int = 20,
        max_steps: int = 1,
        retrieval_depth: int = 10,
    ) -> None:
        if top_k < 1 or max_steps < 1:
            raise ValueError("top_k and max_steps must be at least 1")
        self.name = name
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.reranker = reranker
        self.rerank_depth = max(rerank_depth, top_k)
        self.max_steps = max_steps
        self.retrieval_depth = max(retrieval_depth, top_k)
        self._docs: Mapping[str, Document] = {}

    @property
    def agentic(self) -> bool:
        return self.max_steps > 1

    def index(self, corpus: Sequence[Document]) -> None:
        self._docs = {doc.doc_id: doc for doc in corpus}
        self.retriever.index(corpus)
        if self.reranker is not None:
            self.reranker.fit(corpus)

    def _retrieve(self, text: str, trace: QueryTrace) -> list[str]:
        start = time.perf_counter()
        depth = max(self.retrieval_depth, self.rerank_depth if self.reranker else 0)
        hits = self.retriever.search(text, depth)
        trace.stage_ms["retrieve"] += (time.perf_counter() - start) * 1000
        trace.embedding_tokens.update(self.retriever.embedding_usage(text))
        if self.reranker is not None:
            start = time.perf_counter()
            head = self.reranker.rerank(text, hits[: self.rerank_depth], self._docs)
            hits = head + hits[self.rerank_depth :]
            trace.stage_ms["rerank"] += (time.perf_counter() - start) * 1000
        return [hit.doc_id for hit in hits]

    def _record(self, completion: Completion, trace: QueryTrace) -> None:
        trace.usage.extend(completion.usage)
        trace.stage_ms["generate"] += completion.latency_ms
        trace.latency_modeled |= completion.latency_modeled

    def run(self, query: Query) -> QueryTrace:
        trace = QueryTrace(query.query_id, ranking=[], contexts=[], answer="")
        first = self._retrieve(query.text, trace)
        contexts = first[: self.top_k]
        for _ in range(self.max_steps - 1):
            step = self.generator.next_search(
                query.text, [self._docs[d] for d in contexts], trace.searches
            )
            self._record(step, trace)
            if not step.text:
                break
            trace.searches.append(step.text)
            hop = self._retrieve(step.text, trace)
            contexts += [d for d in hop[: self.top_k] if d not in contexts]
        answer = self.generator.answer(
            query.text, [self._docs[d] for d in contexts], trace.searches
        )
        self._record(answer, trace)
        trace.contexts = contexts
        # Ranking for retrieval metrics: passages in the order the generator received them,
        # then the rest of the first-pass ranking. Single-step pipelines keep their ranking.
        trace.ranking = contexts + [d for d in first if d not in contexts]
        trace.answer = answer.text
        return trace
