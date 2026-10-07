"""Experiment configuration: a YAML file describing a dataset and a matrix of pipelines.

Relative paths are resolved against the current working directory. Pipelines are listed
explicitly under ``pipelines`` and/or generated as the Cartesian product of ``matrix`` axes;
both are merged over ``defaults``.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmbedderConfig(_Strict):
    kind: Literal["tfidf-svd", "hashing", "sentence-transformers", "openai"] = "tfidf-svd"
    dim: int = Field(128, ge=2)
    model: str | None = None

    @property
    def label(self) -> str:
        return {"tfidf-svd": "lsa", "hashing": "hash"}.get(self.kind, self.kind)


class RetrieverConfig(_Strict):
    kind: Literal["bm25", "dense", "hybrid"] = "bm25"
    embedder: EmbedderConfig = EmbedderConfig()
    components: list[RetrieverConfig] = Field(default_factory=list)
    rrf_k: int = Field(60, ge=1)
    weights: list[float] | None = None
    k1: float = 1.2
    b: float = 0.75

    @model_validator(mode="after")
    def _hybrid_components(self) -> RetrieverConfig:
        if self.kind == "hybrid" and len(self.components) == 1:
            raise ValueError("hybrid retriever needs at least two components")
        return self

    def resolved_components(self) -> list[RetrieverConfig]:
        if self.components:
            return self.components
        return [RetrieverConfig(kind="bm25"), RetrieverConfig(kind="dense")]

    @property
    def label(self) -> str:
        if self.kind == "dense":
            return f"dense-{self.embedder.label}"
        if self.kind == "hybrid":
            return "hybrid"
        return "bm25"


class RerankerConfig(_Strict):
    kind: Literal["query-likelihood", "cross-encoder"] = "query-likelihood"
    depth: int = Field(20, ge=1)
    model: str | None = None


class GeneratorConfig(_Strict):
    provider: Literal["mock", "anthropic", "openai"] = "mock"
    model: str | None = None
    price_as: str | None = None
    max_tokens: int = Field(256, ge=1)


class JudgeConfig(_Strict):
    provider: Literal["heuristic", "anthropic", "openai"] = "heuristic"
    model: str | None = None


class PipelineConfig(_Strict):
    name: str
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig | None = None
    generator: GeneratorConfig = GeneratorConfig()
    top_k: int = Field(5, ge=1)
    max_steps: int = Field(1, ge=1, description="1 = single-shot RAG; >1 = agentic loop")


class DatasetConfig(_Strict):
    kind: Literal["synthetic", "beir"] = "synthetic"
    path: Path | None = None
    split: str = "test"
    seed: int = 7
    limit: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def _beir_needs_path(self) -> DatasetConfig:
        if self.kind == "beir" and self.path is None:
            raise ValueError("dataset.path is required for kind=beir")
        return self


class BudgetConfig(_Strict):
    max_cost_per_1k_usd: float | None = None
    max_p95_latency_ms: float | None = None


class ExperimentConfig(_Strict):
    name: str
    dataset: DatasetConfig = DatasetConfig()
    prices: Path = Path("configs/prices.yaml")
    output_dir: Path = Path("runs")
    cache_dir: Path = Path(".rag_eval_cache")
    metrics_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    quality_metric: str = "answer_f1"
    baseline: str | None = None
    budget: BudgetConfig = BudgetConfig()
    bootstrap_samples: int = Field(2000, ge=100)
    seed: int = 0
    judge: JudgeConfig = JudgeConfig()
    pipelines: list[PipelineConfig]

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        names = [p.name for p in self.pipelines]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate pipeline names: {duplicates}")
        if not self.pipelines:
            raise ValueError("experiment defines no pipelines")
        if self.baseline is not None and self.baseline not in names:
            raise ValueError(f"baseline {self.baseline!r} is not a pipeline name")
        return self

    @property
    def baseline_name(self) -> str:
        return self.baseline or self.pipelines[0].name


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def auto_name(pipeline: PipelineConfig) -> str:
    parts = [pipeline.retriever.label]
    if pipeline.reranker is not None:
        parts.append("rerank")
    if pipeline.max_steps > 1:
        parts.append(f"agent{pipeline.max_steps}")
    parts.append(f"k{pipeline.top_k}")
    return "-".join(parts)


def expand_pipelines(raw: dict[str, Any]) -> list[dict[str, Any]]:
    defaults: dict[str, Any] = raw.get("defaults") or {}
    entries = [deep_merge(defaults, p) for p in raw.get("pipelines") or []]
    matrix: dict[str, list[Any]] = raw.get("matrix") or {}
    if matrix:
        axes = list(matrix)
        for combo in itertools.product(*(matrix[a] for a in axes)):
            entry = deep_merge(defaults, dict(zip(axes, combo, strict=True)))
            if "name" not in entry:
                entry["name"] = auto_name(PipelineConfig(name="_", **entry))
            entries.append(entry)
    return entries


def parse_config(raw: dict[str, Any]) -> ExperimentConfig:
    body = {k: v for k, v in raw.items() if k not in {"defaults", "matrix", "pipelines"}}
    body["pipelines"] = expand_pipelines(raw)
    return ExperimentConfig.model_validate(body)


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return parse_config(raw)
