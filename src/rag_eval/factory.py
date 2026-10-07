"""Build datasets, retrievers, generators and judges from configuration."""

from __future__ import annotations

from rag_eval.config import (
    DatasetConfig,
    EmbedderConfig,
    GeneratorConfig,
    JudgeConfig,
    PipelineConfig,
    RerankerConfig,
    RetrieverConfig,
)
from rag_eval.datasets import Dataset, load_beir_dir
from rag_eval.datasets.synthetic import build_synthetic_dataset
from rag_eval.generation import ExtractiveGenerator, Generator, LLMGenerator
from rag_eval.generation.llm import CompletionClient
from rag_eval.judge import HeuristicJudge, Judge, LLMJudge
from rag_eval.pipeline import Pipeline
from rag_eval.retrieval import BM25Retriever, Retriever
from rag_eval.retrieval.dense import DenseRetriever
from rag_eval.retrieval.embedders import Embedder, HashingEmbedder, TfidfSvdEmbedder
from rag_eval.retrieval.hybrid import HybridRetriever
from rag_eval.retrieval.rerank import QueryLikelihoodReranker, Reranker


def load_dataset(cfg: DatasetConfig) -> Dataset:
    if cfg.kind == "synthetic":
        dataset = build_synthetic_dataset(cfg.seed)
    else:
        assert cfg.path is not None  # enforced by DatasetConfig validation
        dataset = load_beir_dir(cfg.path, split=cfg.split)
    return dataset.subset(cfg.limit) if cfg.limit else dataset


def build_embedder(cfg: EmbedderConfig, seed: int) -> Embedder:
    if cfg.kind == "tfidf-svd":
        return TfidfSvdEmbedder(dim=cfg.dim, seed=seed)
    if cfg.kind == "hashing":
        return HashingEmbedder(dim=cfg.dim)
    if cfg.kind == "sentence-transformers":  # pragma: no cover - optional extra
        from rag_eval.retrieval.embedders import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(cfg.model or "sentence-transformers/all-MiniLM-L6-v2")
    from rag_eval.retrieval.embedders import OpenAIEmbedder  # pragma: no cover

    return OpenAIEmbedder(cfg.model or "text-embedding-3-small")  # pragma: no cover


def build_retriever(cfg: RetrieverConfig, seed: int) -> Retriever:
    if cfg.kind == "bm25":
        return BM25Retriever(k1=cfg.k1, b=cfg.b)
    if cfg.kind == "dense":
        return DenseRetriever(build_embedder(cfg.embedder, seed))
    components = [build_retriever(c, seed) for c in cfg.resolved_components()]
    return HybridRetriever(components, rrf_k=cfg.rrf_k, weights=cfg.weights)


def build_reranker(cfg: RerankerConfig) -> Reranker:
    if cfg.kind == "query-likelihood":
        return QueryLikelihoodReranker()
    from rag_eval.retrieval.rerank import CrossEncoderReranker  # pragma: no cover

    return CrossEncoderReranker(
        cfg.model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )  # pragma: no cover


def build_client(provider: str, model: str | None) -> CompletionClient:  # pragma: no cover
    from rag_eval.generation.providers import AnthropicClient, OpenAIClient

    if provider == "anthropic":
        return AnthropicClient(model or "claude-haiku-4-5")
    if provider == "openai":
        return OpenAIClient(model or "gpt-4o-mini")
    raise ValueError(f"unknown provider {provider!r}")


def build_generator(cfg: GeneratorConfig) -> Generator:
    if cfg.provider == "mock":
        return ExtractiveGenerator(price_as=cfg.price_as or "claude-haiku-4-5")
    client = build_client(cfg.provider, cfg.model)  # pragma: no cover
    return LLMGenerator(client, cfg.max_tokens, cfg.price_as)  # pragma: no cover


def build_judge(cfg: JudgeConfig) -> Judge:
    if cfg.provider == "heuristic":
        return HeuristicJudge()
    return LLMJudge(build_client(cfg.provider, cfg.model))  # pragma: no cover


def build_pipeline(
    cfg: PipelineConfig, generator: Generator, seed: int, retrieval_depth: int
) -> Pipeline:
    return Pipeline(
        name=cfg.name,
        retriever=build_retriever(cfg.retriever, seed),
        generator=generator,
        top_k=cfg.top_k,
        reranker=build_reranker(cfg.reranker) if cfg.reranker else None,
        rerank_depth=cfg.reranker.depth if cfg.reranker else 0,
        max_steps=cfg.max_steps,
        retrieval_depth=retrieval_depth,
    )
