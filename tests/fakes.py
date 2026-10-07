"""Deterministic stand-ins for API clients, so LLM code paths run without a network."""

from __future__ import annotations

from collections.abc import Callable

from rag_eval.generation.llm import RawCompletion


class ScriptedClient:
    def __init__(self, reply: Callable[[str, str], str], model: str = "fake-model") -> None:
        self._reply = reply
        self._model = model
        self.calls: list[tuple[str, str]] = []

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int) -> RawCompletion:
        self.calls.append((system, user))
        return RawCompletion(self._reply(system, user), len(user) // 4, 7)
