"""Content-addressed response cache for generator and judge calls.

Keys hash everything that determines the response: the component's identity (model and
parameters), the method, the question, and each context passage's id *and* text. Editing a
document therefore invalidates exactly the generations that saw it. Cached completions keep
the latency recorded when they were first produced, so re-running a report from cache does not
make a pipeline look faster than it is.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_eval.datasets import Document
from rag_eval.generation.base import Completion, Generator
from rag_eval.judge import Judge, Verdict


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _context_key(contexts: Sequence[Document]) -> list[list[str]]:
    return [
        [doc.doc_id, hashlib.sha256(doc.full_text.encode()).hexdigest()[:16]] for doc in contexts
    ]


class ResponseCache:
    """SQLite-backed key/value store. Writes are committed in batches (and on close) because a
    commit per call dominates run time on slow or networked filesystems."""

    def __init__(self, path: Path, commit_every: int = 256) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()
        self.commit_every = commit_every
        self._uncommitted = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        value: dict[str, Any] = json.loads(row[0])
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self._uncommitted += 1
        if self._uncommitted >= self.commit_every:
            self.flush()

    def flush(self) -> None:
        self._conn.commit()
        self._uncommitted = 0

    def close(self) -> None:
        self.flush()
        self._conn.close()


class CachedGenerator:
    def __init__(self, inner: Generator, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return self.inner.name

    def answer(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str] = ()
    ) -> Completion:
        key = cache_key(self.name, "answer", question, _context_key(contexts), list(searches))
        if (hit := self.cache.get(key)) is not None:
            return Completion.from_dict(hit)
        completion = self.inner.answer(question, contexts, searches)
        self.cache.put(key, completion.to_dict())
        return completion

    def next_search(
        self, question: str, contexts: Sequence[Document], searches: Sequence[str]
    ) -> Completion:
        key = cache_key(self.name, "search", question, _context_key(contexts), list(searches))
        if (hit := self.cache.get(key)) is not None:
            return Completion.from_dict(hit)
        completion = self.inner.next_search(question, contexts, searches)
        self.cache.put(key, completion.to_dict())
        return completion


class CachedJudge:
    def __init__(self, inner: Judge, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return self.inner.name

    def judge(
        self,
        question: str,
        answer: str,
        references: Sequence[str],
        contexts: Sequence[Document],
    ) -> Verdict:
        key = cache_key(
            self.name, "judge", question, answer, list(references), _context_key(contexts)
        )
        if (hit := self.cache.get(key)) is not None:
            return Verdict.from_dict(hit)
        verdict = self.inner.judge(question, answer, references, contexts)
        self.cache.put(key, verdict.to_dict())
        return verdict
