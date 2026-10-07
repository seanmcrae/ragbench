"""Deterministic extractive generator: the offline default that needs no API key.

It scores context sentences by query-term overlap, extracts the answer span (the predicate
after the first copula, or the run of non-query terms), and cites the source passage. For
agentic mode it plans follow-up searches by substituting a resolved bridge fact into the
query, a heuristic stand-in for LLM query rewriting. Token usage is computed on the same
prompt an API model would receive, and latency is modeled from those token counts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval.cost import TokenUsage
from rag_eval.datasets import Document
from rag_eval.generation.base import Completion
from rag_eval.generation.prompts import (
    ABSTAIN,
    ANSWER_SYSTEM,
    SEARCH_SYSTEM,
    answer_prompt,
    search_prompt,
)
from rag_eval.text import STOPWORDS, content_terms, count_tokens, normalize_term, split_sentences

_COPULA = re.compile(r"\s(?:is|are)\s")
_WORDS = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Illustrative serving latency: fixed overhead plus prefill and decode time per token."""

    base_ms: float = 350.0
    per_input_token_ms: float = 0.08
    per_output_token_ms: float = 15.0

    def estimate(self, input_tokens: int, output_tokens: int) -> float:
        return (
            self.base_ms
            + self.per_input_token_ms * input_tokens
            + self.per_output_token_ms * output_tokens
        )


@dataclass(frozen=True, slots=True)
class _Sentence:
    doc_id: str
    text: str
    terms: frozenset[str]


def _ordered_unique(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def extract_span(sentence: str, query_terms: frozenset[str]) -> str:
    """Answer span from a fact sentence.

    Help-center facts are mostly copular ("The sync interval is every 15 minutes."), so the
    predicate after the first "is"/"are" is the answer. Otherwise fall back to the shortest
    window covering every content word the query did not already contain.
    """
    body = sentence.strip().rstrip(".")
    match = _COPULA.search(body)
    if match:
        return body[match.end() :].strip()
    words = _WORDS.findall(body)
    novel = [
        i
        for i, w in enumerate(words)
        if (t := w.lower().strip(",.;:()")) not in STOPWORDS
        and normalize_term(t) not in query_terms
    ]
    return " ".join(words[novel[0] : novel[-1] + 1]) if novel else body


class ExtractiveGenerator:
    def __init__(
        self,
        price_as: str = "claude-haiku-4-5",
        min_coverage: float = 0.34,
        stop_coverage: float = 0.75,
        latency: LatencyModel | None = None,
    ) -> None:
        self.price_as = price_as
        self.min_coverage = min_coverage
        self.stop_coverage = stop_coverage
        self.latency = latency or LatencyModel()

    @property
    def name(self) -> str:
        return (
            f"extractive(price_as={self.price_as},min={self.min_coverage},"
            f"stop={self.stop_coverage})"
        )

    def _completion(self, system: str, user: str, text: str) -> Completion:
        usage = TokenUsage(self.price_as, count_tokens(system + "\n" + user), count_tokens(text))
        latency = self.latency.estimate(usage.input_tokens, usage.output_tokens)
        return Completion(text, (usage,), latency, latency_modeled=True)

    @staticmethod
    def _sentences(contexts: Sequence[Document]) -> list[_Sentence]:
        return [
            _Sentence(doc.doc_id, s, frozenset(content_terms(s)))
            for doc in contexts
            for s in split_sentences(doc.text)
        ]

    @staticmethod
    def _best(sentences: Sequence[_Sentence], terms: frozenset[str]) -> tuple[_Sentence, int]:
        # max() keeps the first maximum, so ties go to the higher-ranked passage.
        best = max(sentences, key=lambda s: len(terms & s.terms))
        return best, len(terms & best.terms)

    def answer(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str] = ()
    ) -> Completion:
        user = answer_prompt(question, contexts, searches)
        target = frozenset(content_terms(searches[-1] if searches else question))
        sentences = self._sentences(contexts)
        if not sentences or not target:
            return self._completion(ANSWER_SYSTEM, user, ABSTAIN)
        best, overlap = self._best(sentences, target)
        if overlap / len(target) < self.min_coverage:
            return self._completion(ANSWER_SYSTEM, user, ABSTAIN)
        text = f"{extract_span(best.text, target)} [{best.doc_id}]"
        return self._completion(ANSWER_SYSTEM, user, text)

    def next_search(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str]
    ) -> Completion:
        user = search_prompt(question, contexts, searches)
        q_terms = _ordered_unique(content_terms(question))
        q_set = frozenset(q_terms)
        sentences = self._sentences(contexts)
        if not sentences or not q_terms:
            return self._completion(SEARCH_SYSTEM, user, "")
        best, overlap = self._best(sentences, q_set)
        if overlap / len(q_set) >= self.stop_coverage:
            return self._completion(SEARCH_SYSTEM, user, "")
        # The bridge is the sentence that best explains what the best answer candidate misses;
        # its resolved terms replace the question terms it satisfies.
        uncovered = q_set - best.terms
        candidates = [s for s in sentences if s is not best]
        if not candidates:
            return self._completion(SEARCH_SYSTEM, user, "")
        bridge, bridge_overlap = self._best(candidates, uncovered)
        if bridge_overlap == 0:
            return self._completion(SEARCH_SYSTEM, user, "")
        resolved = [t for t in q_terms if t not in bridge.terms]
        novel = [t for t in _ordered_unique(content_terms(bridge.text)) if t not in q_set]
        query = " ".join(resolved + novel)
        if query in searches:
            return self._completion(SEARCH_SYSTEM, user, "")
        return self._completion(SEARCH_SYSTEM, user, query)
