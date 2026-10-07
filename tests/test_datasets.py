from pathlib import Path

import pytest

from rag_eval.datasets import Dataset, Document, Query, load_beir_dir, save_beir_dir


def test_round_trip_preserves_content(tmp_path: Path, tiny_dataset: Dataset) -> None:
    save_beir_dir(tiny_dataset, tmp_path)
    loaded = load_beir_dir(tmp_path, name="tiny")
    assert loaded.corpus == tiny_dataset.corpus
    assert loaded.qrels == tiny_dataset.qrels
    assert loaded.answers == tiny_dataset.answers
    assert loaded.fingerprint() == tiny_dataset.fingerprint()


def test_loader_keeps_only_queries_judged_in_split(tmp_path: Path, tiny_dataset: Dataset) -> None:
    save_beir_dir(tiny_dataset, tmp_path)
    (tmp_path / "qrels" / "dev.tsv").write_text("query-id\tcorpus-id\tscore\nq2\td4\t1\n")
    loaded = load_beir_dir(tmp_path, split="dev")
    assert set(loaded.queries) == {"q2"}
    assert set(loaded.answers) == {"q2"}


def test_qrels_must_reference_known_ids() -> None:
    corpus = {"d1": Document("d1", "", "text")}
    with pytest.raises(ValueError, match="unknown documents"):
        Dataset("bad", corpus, {"q1": Query("q1", "x")}, {"q1": {"missing": 1}})
    with pytest.raises(ValueError, match="unknown queries"):
        Dataset("bad", corpus, {}, {"q1": {"d1": 1}})


def test_subset_and_fingerprint_change(tiny_dataset: Dataset) -> None:
    small = tiny_dataset.subset(2)
    assert [q.query_id for q in small.evaluable_queries()] == ["q1", "q2"]
    assert small.fingerprint() != tiny_dataset.fingerprint()
