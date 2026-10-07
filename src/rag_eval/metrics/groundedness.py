"""Offline groundedness heuristic based on citation coverage.

Two signals, both in [0, 1]:
  citation_rate  share of answer sentences that cite at least one retrieved document
  support        share of the answer's content terms that occur in the documents it cites

``groundedness`` is their product: an uncited answer scores 0 however true it is, and a cited
answer scores only as high as the cited text actually supports it. This is a lexical proxy;
the LLM judge covers paraphrased support that term overlap misses.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from rag_eval.datasets import Document
from rag_eval.text import content_terms, split_sentences

CITATION = re.compile(r"\[([^\[\]]+)\]")


@dataclass(frozen=True, slots=True)
class Groundedness:
    citation_rate: float
    support: float

    @property
    def score(self) -> float:
        return self.citation_rate * self.support


def extract_citations(text: str) -> list[str]:
    """Doc ids cited as ``[doc-id]`` or ``[a, b]``, in first-seen order."""
    seen: dict[str, None] = {}
    for group in CITATION.findall(text):
        for doc_id in group.split(","):
            seen.setdefault(doc_id.strip(), None)
    return list(seen)


def groundedness(
    answer: str, docs: Mapping[str, Document], retrieved: Collection[str]
) -> Groundedness:
    sentences = split_sentences(answer)
    if not sentences:
        return Groundedness(0.0, 0.0)
    cited_sentences = 0
    cited_ids: set[str] = set()
    for sentence in sentences:
        ids = [d for d in extract_citations(sentence) if d in retrieved and d in docs]
        cited_sentences += bool(ids)
        cited_ids.update(ids)
    terms = content_terms(CITATION.sub(" ", answer))
    if not terms:
        return Groundedness(cited_sentences / len(sentences), 0.0)
    cited_vocab = {t for d in cited_ids for t in content_terms(docs[d].full_text)}
    support = sum(t in cited_vocab for t in terms) / len(terms)
    return Groundedness(cited_sentences / len(sentences), support)
