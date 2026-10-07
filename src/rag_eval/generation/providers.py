"""Anthropic and OpenAI completion clients (optional extras; keys come from the environment)."""

from __future__ import annotations

import os

from rag_eval.generation.llm import RawCompletion


def _require_env(var: str) -> None:
    if not os.environ.get(var):
        raise RuntimeError(f"{var} is not set; export it or use the offline mock provider")


class AnthropicClient:  # pragma: no cover - requires network and an API key
    def __init__(self, model: str = "claude-haiku-4-5", temperature: float = 0.0) -> None:
        _require_env("ANTHROPIC_API_KEY")
        import anthropic

        self._client = anthropic.Anthropic()
        self._model = model
        self.temperature = temperature

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int) -> RawCompletion:
        response = self._client.messages.create(
            model=self._model,
            system=system,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return RawCompletion(text, response.usage.input_tokens, response.usage.output_tokens)


class OpenAIClient:  # pragma: no cover - requires network and an API key
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        _require_env("OPENAI_API_KEY")
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self.temperature = temperature

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, system: str, user: str, max_tokens: int) -> RawCompletion:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = response.usage
        return RawCompletion(
            response.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
