from pathlib import Path

import pytest

from rag_eval.cost import ModelPrice, PriceTable, TokenUsage, cost_per_1k_queries

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def table() -> PriceTable:
    return PriceTable(
        as_of="2026-01-01",
        currency="USD",
        models={"llm": ModelPrice(3.0, 15.0), "embed": ModelPrice(0.02, 0.0)},
    )


def test_call_cost_hand_computed(table: PriceTable) -> None:
    # 2000 input tokens at $3/M + 100 output tokens at $15/M = 0.006 + 0.0015
    assert table.call_cost(TokenUsage("llm", 2000, 100)) == pytest.approx(0.0075)


def test_query_cost_sums_calls_and_embeddings(table: PriceTable) -> None:
    calls = [TokenUsage("llm", 1000, 50), TokenUsage("llm", 500, 0)]
    cost = table.query_cost(calls, {"embed": 1_000_000})
    assert cost == pytest.approx(0.003 + 0.00075 + 0.0015 + 0.02)


def test_unknown_model_is_a_clear_error(table: PriceTable) -> None:
    with pytest.raises(ValueError, match="known models: embed, llm"):
        table.call_cost(TokenUsage("mystery", 1, 1))


def test_cost_per_1k() -> None:
    assert cost_per_1k_queries([0.001, 0.003]) == pytest.approx(2.0)
    assert cost_per_1k_queries([]) == 0.0


def test_bundled_price_table_loads() -> None:
    table = PriceTable.load(ROOT / "configs" / "prices.yaml")
    assert table.as_of == "2026-10-07"
    assert table.price("claude-sonnet-4-5") == ModelPrice(3.0, 15.0)


def test_rejects_other_units(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("as_of: x\nunit: per_token\nmodels: {}\n")
    with pytest.raises(ValueError, match="unit"):
        PriceTable.load(path)
