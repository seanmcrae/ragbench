import pytest

from rag_eval.config import BudgetConfig
from rag_eval.pareto import Candidate, pareto_frontier, recommend
from rag_eval.stats import paired_bootstrap


def test_identical_samples_have_zero_width_interval() -> None:
    ci = paired_bootstrap([0.2, 0.4, 0.6], [0.2, 0.4, 0.6])
    assert (ci.delta, ci.low, ci.high) == (0.0, 0.0, 0.0)
    assert not ci.significant
    assert ci.p_value == 1.0


def test_constant_improvement_is_significant() -> None:
    a = [0.1 * i for i in range(20)]
    ci = paired_bootstrap(a, [x + 0.1 for x in a], n_resamples=500, seed=3)
    assert ci.delta == pytest.approx(0.1)
    assert ci.low == pytest.approx(0.1)
    assert ci.high == pytest.approx(0.1)
    assert ci.significant
    assert ci.p_value == pytest.approx(1 / 500)


def test_pairing_beats_unpaired_noise() -> None:
    # Large between-query variance, small consistent per-query gain: a paired test sees it.
    a = [0.0, 1.0] * 15
    b = [x + 0.05 for x in a]
    assert paired_bootstrap(a, b, seed=1).significant


def test_noisy_difference_ci_brackets_zero_and_is_seeded() -> None:
    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    first = paired_bootstrap(a, b, seed=7)
    assert first.low < 0 < first.high
    assert paired_bootstrap(a, b, seed=7) == first


def test_regression_gate_needs_significance_and_size() -> None:
    a = [1.0] * 30
    worse = paired_bootstrap(a, [0.9] * 30)
    assert worse.is_regression(tolerance=0.05)
    assert not worse.is_regression(tolerance=0.2)
    noisy = paired_bootstrap([1.0, 0.0] * 5, [0.0, 1.0] * 5)
    assert not noisy.is_regression()


@pytest.mark.parametrize(
    ("a", "b", "kwargs"),
    [([1.0], [1.0, 2.0], {}), ([], [], {}), ([1.0], [1.0], {"confidence": 1.5})],
)
def test_bootstrap_validates_input(
    a: list[float], b: list[float], kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, **kwargs)  # type: ignore[arg-type]


CANDIDATES = [
    Candidate("cheap", 0.50, 0.40, 800),
    Candidate("balanced", 0.60, 0.50, 850),
    Candidate("dominated", 0.55, 0.60, 900),
    Candidate("premium", 0.65, 2.00, 1900),
    Candidate("twin", 0.60, 0.50, 850),
]


def test_frontier_excludes_dominated_and_keeps_ties() -> None:
    names = [c.name for c in pareto_frontier(CANDIDATES)]
    assert names == ["cheap", "balanced", "twin", "premium"]


def test_recommend_respects_budget_and_breaks_ties() -> None:
    rec = recommend(CANDIDATES, BudgetConfig(max_cost_per_1k_usd=1.0, max_p95_latency_ms=1500))
    assert rec.chosen is not None and rec.chosen.name == "balanced"
    assert set(rec.rejected) == {"premium"}
    assert "cost $2.000/1k > $1.00" in rec.rejected["premium"]
    assert "p95 1900 ms > 1500 ms" in rec.rejected["premium"]


def test_recommend_without_budget_picks_best_quality() -> None:
    chosen = recommend(CANDIDATES, BudgetConfig()).chosen
    assert chosen is not None and chosen.name == "premium"


def test_recommend_with_impossible_budget() -> None:
    rec = recommend(CANDIDATES, BudgetConfig(max_cost_per_1k_usd=0.01))
    assert rec.chosen is None
    assert rec.frontier == []
    assert len(rec.rejected) == len(CANDIDATES)
