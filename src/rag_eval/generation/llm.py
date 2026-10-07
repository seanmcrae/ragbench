"""Generator and client abstractions for hosted LLM APIs."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval.cost import TokenUsage
from rag_eval.datasets import Document
from rag_eval.generation.base import Completion
from rag_eval.generation.prompts import ANSWER_SYSTEM, SEARCH_SYSTEM, answer_prompt, search_prompt


@dataclass(frozen=True, slots=True)
class RawCompletion:
    text: str
    input_tokens: int
    output_tokens: int


class CompletionClient(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, system: str, user: str, max_tokens: int) -> RawCompletion: ...


def parse_search_reply(reply: str) -> str:
    """Map a planner reply to a follow-up query, or "" when the model says it is done."""
    text = reply.strip()
    if not text or text.upper().startswith("DONE"):
        return ""
    if text.upper().startswith("SEARCH:"):
        text = text[len("SEARCH:") :]
    return text.strip().splitlines()[0].strip() if text.strip() else ""


class LLMGenerator:
    def __init__(
        self, client: CompletionClient, max_tokens: int = 256, price_as: str | None = None
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.price_as = price_as or client.model

    @property
    def name(self) -> str:
        return f"llm({self.client.provider}:{self.client.model},max_tokens={self.max_tokens})"

    def _call(self, system: str, user: str) -> tuple[str, Completion]:
        start = time.perf_counter()
        raw = self.client.complete(system, user, self.max_tokens)
        elapsed = (time.perf_counter() - start) * 1000
        usage = TokenUsage(self.price_as, raw.input_tokens, raw.output_tokens)
        return raw.text, Completion(raw.text.strip(), (usage,), elapsed)

    def answer(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str] = ()
    ) -> Completion:
        _, completion = self._call(ANSWER_SYSTEM, answer_prompt(question, contexts, searches))
        return completion

    def next_search(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str]
    ) -> Completion:
        raw, completion = self._call(SEARCH_SYSTEM, search_prompt(question, contexts, searches))
        return Completion(parse_search_reply(raw), completion.usage, completion.latency_ms, False)
