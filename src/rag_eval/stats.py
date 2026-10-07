"""Paired bootstrap confidence intervals for metric deltas between two pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DeltaCI:
    """Mean of B minus mean of A over the same queries, with a percentile bootstrap CI."""

    mean_a: float
    mean_b: float
    low: float
    high: float
    p_value: float
    n: int
    confidence: float

    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a

    @property
    def significant(self) -> bool:
        return self.low > 0 or self.high < 0

    def is_regression(self, tolerance: float = 0.0) -> bool:
        """Release-gate rule: B is confidently worse than A by more than ``tolerance``.

        Both conditions are required: the CI must exclude zero (the drop is not noise) and the
        point estimate must exceed the tolerance (the drop matters).
        """
        return self.high < 0 and self.delta < -tolerance


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> DeltaCI:
    """Resample queries with replacement, keeping each query's A and B values together.

    Pairing removes between-query variance (some questions are simply harder), which is what
    makes small evaluation sets usable. The p-value is the two-sided bootstrap estimate
    2 * min(P(delta* <= 0), P(delta* >= 0)), floored at 1 / n_resamples.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    if not a:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    resampled = diffs[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(resampled, [alpha, 1 - alpha])
    tail = min(float(np.mean(resampled <= 0)), float(np.mean(resampled >= 0)))
    p_value = min(1.0, max(2 * tail, 1 / n_resamples))
    return DeltaCI(
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        low=float(low),
        high=float(high),
        p_value=p_value,
        n=len(diffs),
        confidence=confidence,
    )
