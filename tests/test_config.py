from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rag_eval.config import auto_name, deep_merge, load_config, parse_config
from rag_eval.factory import build_retriever, load_dataset
from rag_eval.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent


def base(**extra: Any) -> dict[str, Any]:
    return {"name": "t", "pipelines": [{"name": "a"}], **extra}


def test_deep_merge_does_not_mutate_inputs() -> None:
    left: dict[str, Any] = {"generator": {"provider": "mock", "price_as": "x"}, "top_k": 5}
    right = {"generator": {"price_as": "y"}}
    assert deep_merge(left, right) == {
        "generator": {"provider": "mock", "price_as": "y"},
        "top_k": 5,
    }
    assert left["generator"]["price_as"] == "x"


def test_defaults_apply_to_explicit_and_matrix_pipelines() -> None:
    cfg = parse_config(
        {
            "name": "t",
            "defaults": {"top_k": 7, "generator": {"price_as": "gpt-4o-mini"}},
            "pipelines": [{"name": "explicit", "top_k": 2}],
            "matrix": {"retriever": [{"kind": "bm25"}, {"kind": "hybrid"}], "max_steps": [1, 2]},
        }
    )
    names = [p.name for p in cfg.pipelines]
    assert names == ["explicit", "bm25-k7", "bm25-agent2-k7", "hybrid-k7", "hybrid-agent2-k7"]
    assert cfg.pipelines[0].top_k == 2
    assert all(p.generator.price_as == "gpt-4o-mini" for p in cfg.pipelines)
    assert cfg.baseline_name == "explicit"


def test_auto_name_covers_every_dimension() -> None:
    cfg = parse_config(
        {
            "name": "t",
            "matrix": {
                "retriever": [{"kind": "dense", "embedder": {"kind": "hashing"}}],
                "reranker": [{"kind": "query-likelihood"}],
                "top_k": [3],
            },
        }
    )
    assert auto_name(cfg.pipelines[0]) == "dense-hash-rerank-k3"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (base(pipelines=[{"name": "a"}, {"name": "a"}]), "duplicate pipeline names"),
        (base(pipelines=[]), "no pipelines"),
        (base(baseline="missing"), "baseline"),
        (base(dataset={"kind": "beir"}), "dataset.path"),
        (base(pipelines=[{"name": "a", "top_k": 0}]), "greater than or equal to 1"),
        (base(pipelines=[{"name": "a", "retriever": {"kind": "bm42"}}]), "bm25"),
        (base(unknown_key=1), "Extra inputs"),
        (
            base(pipelines=[{"name": "a", "retriever": {"kind": "hybrid", "components": [{}]}}]),
            "at least two components",
        ),
    ],
)
def test_invalid_configs_fail_loudly(raw: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_config(raw)


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_bundled_configs_parse() -> None:
    example = load_config(ROOT / "configs" / "example.yaml")
    assert example.baseline_name == "bm25-k5"
    assert len(example.pipelines) == 8
    scifact = load_config(ROOT / "configs" / "scifact.yaml")
    assert scifact.dataset.kind == "beir"


def test_factory_builds_default_hybrid_and_limits_dataset() -> None:
    cfg = parse_config(base(pipelines=[{"name": "h", "retriever": {"kind": "hybrid"}}]))
    retriever = build_retriever(cfg.pipelines[0].retriever, seed=0)
    assert isinstance(retriever, HybridRetriever)
    assert retriever.name == "hybrid[bm25+dense[tfidf-svd-128]]"
    limited = parse_config(base(dataset={"kind": "synthetic", "limit": 4}))
    assert len(load_dataset(limited.dataset).queries) == 4
