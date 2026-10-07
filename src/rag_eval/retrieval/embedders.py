"""Pluggable text embedders.

The two offline embedders are deterministic and need no downloads, so they are the defaults.
Sentence-transformers and OpenAI embedders are optional extras imported lazily.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from rag_eval.text import content_terms

Matrix = NDArray[np.float32]


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def price_as(self) -> str | None:
        """Model name in the price table to bill query embeddings against, if any."""

    def fit(self, texts: Sequence[str]) -> None: ...

    def embed(self, texts: Sequence[str]) -> Matrix:
        """L2-normalised row vectors, one per text."""


def l2_normalize(matrix: NDArray[np.floating]) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: Matrix = (matrix / norms).astype(np.float32)
    return normalized


class TfidfSvdEmbedder:
    """Latent semantic embeddings: TF-IDF over unigrams and bigrams reduced with truncated SVD."""

    def __init__(self, dim: int = 128, seed: int = 0) -> None:
        self.dim = dim
        self.seed = seed
        self._vectorizer = TfidfVectorizer(
            tokenizer=content_terms,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._svd: TruncatedSVD | None = None

    @property
    def name(self) -> str:
        return f"tfidf-svd-{self.dim}"

    @property
    def price_as(self) -> str | None:
        return None

    def fit(self, texts: Sequence[str]) -> None:
        tfidf = self._vectorizer.fit_transform(texts)
        n_components = max(1, min(self.dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=self.seed)
        self._svd.fit(tfidf)

    def embed(self, texts: Sequence[str]) -> Matrix:
        if self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder.embed called before fit")
        return l2_normalize(self._svd.transform(self._vectorizer.transform(texts)))


class HashingEmbedder:
    """Stateless character n-gram hashing; robust to morphology, needs no fitting."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=dim,
            alternate_sign=False,
            norm=None,
        )

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    @property
    def price_as(self) -> str | None:
        return None

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> Matrix:
        counts = self._vectorizer.transform([t.lower() for t in texts])
        return l2_normalize(np.log1p(counts.toarray()))


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model)

    @property
    def name(self) -> str:
        return f"st:{self.model_name}"

    @property
    def price_as(self) -> str | None:
        return None

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> Matrix:
        vectors = self._model.encode(list(texts), batch_size=64, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors))


class OpenAIEmbedder:  # pragma: no cover - optional dependency, needs network
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 256) -> None:
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model_name = model
        self.batch_size = batch_size
        self._client = OpenAI()

    @property
    def name(self) -> str:
        return f"openai:{self.model_name}"

    @property
    def price_as(self) -> str | None:
        return self.model_name

    def fit(self, texts: Sequence[str]) -> None:
        return None

    def embed(self, texts: Sequence[str]) -> Matrix:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            rows.extend(item.embedding for item in response.data)
        return l2_normalize(np.asarray(rows))
