"""Answer-quality judges sharing one rubric: a deterministic mock and an LLM-as-judge."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_eval.cost import TokenUsage
from rag_eval.datasets import Document
from rag_eval.generation.llm import CompletionClient
from rag_eval.generation.prompts import format_passages
from rag_eval.metrics.answer import token_f1
from rag_eval.metrics.groundedness import CITATION, groundedness

RUBRIC = """Score the ANSWER from 1 to 5.
5: correct and complete, and every claim is supported by a cited passage.
4: correct, with minor omissions or extra words; support is present.
3: partially correct, or correct but weakly supported by the cited passages.
2: mostly incorrect, though related to the question.
1: wrong, unsupported, or an abstention when the passages contain the answer."""

JUDGE_SYSTEM = (
    "You are a strict evaluator of answers produced by a retrieval-augmented system.\n"
    f"{RUBRIC}\n"
    'Reply with one line "Score: N" followed by a one-sentence rationale.'
)

_SCORE = re.compile(r"Score:\s*([1-5])")


@dataclass(frozen=True)
class Verdict:
    score: int
    rationale: str
    usage: tuple[TokenUsage, ...] = field(default_factory=tuple)
    latency_ms: float = 0.0

    @property
    def normalized(self) -> float:
        """Rubric score mapped onto [0, 1]."""
        return (self.score - 1) / 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "rationale": self.rationale,
            "usage": [[u.model, u.input_tokens, u.output_tokens] for u in self.usage],
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Verdict:
        return cls(
            int(raw["score"]),
            raw["rationale"],
            tuple(TokenUsage(m, int(i), int(o)) for m, i, o in raw["usage"]),
            float(raw["latency_ms"]),
        )


class Judge(Protocol):
    @property
    def name(self) -> str: ...

    def judge(
        self,
        question: str,
        answer: str,
        references: Sequence[str],
        contexts: Sequence[Document],
    ) -> Verdict: ...


class HeuristicJudge:
    """Applies the rubric mechanically from token F1 and citation-coverage groundedness.

    It is the offline default so pipelines can be compared without keys. Without reference
    answers (e.g. BEIR SciFact) it can only grade support, so scores cap at 3.
    """

    @property
    def name(self) -> str:
        return "heuristic-rubric-v1"

    def judge(
        self,
        question: str,
        answer: str,
        references: Sequence[str],
        contexts: Sequence[Document],
    ) -> Verdict:
        docs = {d.doc_id: d for d in contexts}
        grounded = groundedness(answer, docs, docs.keys()).score
        if not references:
            score = 3 if grounded >= 0.5 else 1
            return Verdict(score, f"no reference; groundedness {grounded:.2f}")
        f1 = token_f1(CITATION.sub(" ", answer), references)
        if f1 >= 0.8:
            score = 5
        elif f1 >= 0.5:
            score = 4
        elif f1 >= 0.25:
            score = 3
        elif f1 > 0:
            score = 2
        else:
            score = 1
        if grounded < 0.5 and score > 1:
            score -= 1
        return Verdict(score, f"token F1 {f1:.2f}, groundedness {grounded:.2f}")


def parse_verdict(reply: str) -> tuple[int, str]:
    match = _SCORE.search(reply)
    if not match:
        raise ValueError(f"judge reply has no 'Score: N' line: {reply[:80]!r}")
    rationale = reply[match.end() :].strip(" .\n") or reply.strip()
    return int(match.group(1)), rationale


class LLMJudge:
    """LLM-as-judge with the shared rubric. Known biases (verbosity, self-preference) apply;
    use a different model family from the generator and spot-check against human labels."""

    def __init__(
        self, client: CompletionClient, max_tokens: int = 120, price_as: str | None = None
    ) -> None:
        self.client = client
        self.max_tokens = max_tokens
        self.price_as = price_as or client.model

    @property
    def name(self) -> str:
        return f"llm-judge({self.client.provider}:{self.client.model})"

    def judge(
        self,
        question: str,
        answer: str,
        references: Sequence[str],
        contexts: Sequence[Document],
    ) -> Verdict:
        user = (
            f"QUESTION: {question}\n\nPASSAGES:\n{format_passages(contexts)}\n\n"
            f"REFERENCE ANSWERS: {' | '.join(references) or '(none)'}\n\nANSWER: {answer}"
        )
        start = time.perf_counter()
        raw = self.client.complete(JUDGE_SYSTEM, user, self.max_tokens)
        elapsed = (time.perf_counter() - start) * 1000
        score, rationale = parse_verdict(raw.text)
        usage = TokenUsage(self.price_as, raw.input_tokens, raw.output_tokens)
        return Verdict(score, rationale, (usage,), elapsed)
