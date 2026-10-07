"""Prompt templates shared by API generators and the mock's token accounting."""

from __future__ import annotations

from collections.abc import Sequence

from rag_eval.datasets import Document

ANSWER_SYSTEM = (
    "You answer questions about a software product using only the numbered passages provided. "
    "Reply with the shortest complete answer, then cite the supporting passage ids in square "
    'brackets, for example "every 15 minutes [integration-xero-sync]". If the passages do not '
    'contain the answer, reply exactly "I could not find this in the provided documents."'
)

SEARCH_SYSTEM = (
    "You decide whether the passages are enough to answer the question. If they are, reply "
    "exactly DONE. Otherwise reply with one search query, prefixed with SEARCH:, that would "
    "retrieve the missing fact. Use facts from the passages to make the query specific."
)

ABSTAIN = "I could not find this in the provided documents."


def format_passages(contexts: Sequence[Document]) -> str:
    return "\n".join(f"[{doc.doc_id}] {doc.title}: {doc.text}" for doc in contexts)


def answer_prompt(question: str, contexts: Sequence[Document], searches: Sequence[str]) -> str:
    parts = [f"Passages:\n{format_passages(contexts)}"]
    if searches:
        parts.append("Searches already run: " + "; ".join(searches))
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def search_prompt(question: str, contexts: Sequence[Document], searches: Sequence[str]) -> str:
    return answer_prompt(question, contexts, searches)
