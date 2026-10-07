"""Dataset model and on-disk format.

The on-disk layout follows BEIR so public benchmarks load without conversion:

    <dir>/corpus.jsonl          {"_id", "title", "text"}
    <dir>/queries.jsonl         {"_id", "text", "metadata"?}
    <dir>/qrels/<split>.tsv     query-id <TAB> corpus-id <TAB> score   (header row)
    <dir>/answers.jsonl         {"query_id", "answers": [...]}         (optional, not in BEIR)
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Qrels = dict[str, dict[str, int]]
"""query id -> {doc id -> graded relevance}; grades > 0 are relevant."""


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}" if self.title else self.text


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Dataset:
    name: str
    corpus: dict[str, Document]
    queries: dict[str, Query]
    qrels: Qrels
    answers: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown_queries = set(self.qrels) - set(self.queries)
        if unknown_queries:
            raise ValueError(f"qrels reference unknown queries: {sorted(unknown_queries)[:5]}")
        unknown_docs = {d for judged in self.qrels.values() for d in judged} - set(self.corpus)
        if unknown_docs:
            raise ValueError(f"qrels reference unknown documents: {sorted(unknown_docs)[:5]}")

    @property
    def has_answers(self) -> bool:
        return bool(self.answers)

    def evaluable_queries(self) -> list[Query]:
        """Queries with at least one relevant judgement, in stable id order."""
        return [
            self.queries[qid]
            for qid in sorted(self.qrels)
            if any(grade > 0 for grade in self.qrels[qid].values())
        ]

    def subset(self, limit: int) -> Dataset:
        """First ``limit`` evaluable queries; the corpus is kept whole."""
        keep = {q.query_id for q in self.evaluable_queries()[:limit]}
        return Dataset(
            name=self.name,
            corpus=self.corpus,
            queries={qid: q for qid, q in self.queries.items() if qid in keep},
            qrels={qid: r for qid, r in self.qrels.items() if qid in keep},
            answers={qid: a for qid, a in self.answers.items() if qid in keep},
        )

    def fingerprint(self) -> str:
        """Content hash used to key caches and stamp result manifests."""
        digest = hashlib.sha256()
        for doc_id in sorted(self.corpus):
            doc = self.corpus[doc_id]
            digest.update(f"{doc_id}\x1f{doc.title}\x1f{doc.text}\x1e".encode())
        for qid in sorted(self.queries):
            digest.update(f"{qid}\x1f{self.queries[qid].text}\x1e".encode())
        digest.update(json.dumps(self.qrels, sort_keys=True).encode())
        digest.update(json.dumps(self.answers, sort_keys=True).encode())
        return digest.hexdigest()[:16]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_beir_dir(path: Path, split: str = "test", name: str | None = None) -> Dataset:
    """Load a BEIR-format directory, keeping only queries judged in ``split``."""
    corpus = {
        str(row["_id"]): Document(str(row["_id"]), row.get("title", ""), row["text"])
        for row in _read_jsonl(path / "corpus.jsonl")
    }
    qrels: Qrels = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)  # header
        for qid, doc_id, score in reader:
            qrels.setdefault(qid, {})[doc_id] = int(score)
    queries = {
        str(row["_id"]): Query(
            str(row["_id"]),
            row["text"],
            {k: str(v) for k, v in (row.get("metadata") or {}).items()},
        )
        for row in _read_jsonl(path / "queries.jsonl")
        if str(row["_id"]) in qrels
    }
    answers: dict[str, list[str]] = {}
    answers_path = path / "answers.jsonl"
    if answers_path.exists():
        answers = {
            str(row["query_id"]): list(row["answers"])
            for row in _read_jsonl(answers_path)
            if str(row["query_id"]) in qrels
        }
    return Dataset(name or path.name, corpus, queries, qrels, answers)


def save_beir_dir(dataset: Dataset, path: Path, split: str = "test") -> None:
    (path / "qrels").mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        path / "corpus.jsonl",
        ({"_id": d.doc_id, "title": d.title, "text": d.text} for d in dataset.corpus.values()),
    )
    _write_jsonl(
        path / "queries.jsonl",
        (
            {"_id": q.query_id, "text": q.text, "metadata": dict(q.metadata)}
            for q in dataset.queries.values()
        ),
    )
    with (path / "qrels" / f"{split}.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for qid, judged in dataset.qrels.items():
            for doc_id, grade in judged.items():
                writer.writerow([qid, doc_id, grade])
    if dataset.answers:
        _write_jsonl(
            path / "answers.jsonl",
            ({"query_id": qid, "answers": a} for qid, a in dataset.answers.items()),
        )
