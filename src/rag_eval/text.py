"""Tokenisation and normalisation shared by retrievers, metrics and the mock generator."""

from __future__ import annotations

import math
import re
import string

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})

STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    just me more most my no nor not now of off on once only or other our ours out over own
    same she should so some such than that the their theirs them then there these they this
    those through to too under until up very was we were what when where which while who whom
    why will with would you your yours get gets got""".split()
)


def normalize_term(token: str) -> str:
    """Fold simple plurals so that 'invoices' and 'invoice' share a term."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens, no stopword removal."""
    return _WORD.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Index/query terms: tokens minus stopwords, with plural folding."""
    return [normalize_term(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalisation: lower-case, drop punctuation and articles."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def count_tokens(text: str) -> int:
    """Approximate BPE token count without a tokenizer download.

    Each word-like piece costs ceil(len/4) tokens and each punctuation mark one token,
    which tracks cl100k-style tokenisers within roughly 10-15% on English prose. Provider
    adapters report exact usage from the API; this estimate is only used offline.
    """
    pieces = re.findall(r"\w+|[^\w\s]", text)
    return sum(max(1, math.ceil(len(p) / 4)) if p[0].isalnum() else 1 for p in pieces)
