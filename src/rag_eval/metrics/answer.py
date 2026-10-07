"""Answer-quality metrics against reference answers (SQuAD conventions)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from rag_eval.text import normalize_answer


def exact_match(prediction: str, references: Sequence[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(ref) for ref in references))


def _f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Max bag-of-tokens F1 over the references."""
    return max((_f1(prediction, ref) for ref in references), default=0.0)
