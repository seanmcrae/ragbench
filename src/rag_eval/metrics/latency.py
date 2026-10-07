"""Latency summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default); 0.0 for an empty sample."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {"p50": percentile(values, 50), "p95": percentile(values, 95)}
