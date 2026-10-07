from pathlib import Path

from rag_eval.datasets import load_beir_dir
from rag_eval.datasets.synthetic import build_synthetic_dataset

BUNDLED = Path(__file__).resolve().parent.parent / "data" / "synthetic_helpcenter"


def test_generation_is_deterministic() -> None:
    assert build_synthetic_dataset(7).fingerprint() == build_synthetic_dataset(7).fingerprint()
    assert build_synthetic_dataset(7).fingerprint() != build_synthetic_dataset(8).fingerprint()


def test_bundled_files_match_generator() -> None:
    bundled = load_beir_dir(BUNDLED, name="synthetic-helpcenter")
    assert bundled.fingerprint() == build_synthetic_dataset().fingerprint()


def test_size_and_query_mix() -> None:
    ds = build_synthetic_dataset()
    assert 150 <= len(ds.corpus) <= 300
    assert 40 <= len(ds.queries) <= 60
    kinds = {q.metadata["type"] for q in ds.queries.values()}
    assert {"plan_attribute", "error_fix", "policy", "multi_hop"} <= kinds


def test_every_answer_is_stated_in_a_grade_two_document() -> None:
    ds = build_synthetic_dataset()
    for qid, judged in ds.qrels.items():
        assert set(judged.values()) <= {1, 2}
        answer_docs = [ds.corpus[d].text for d, grade in judged.items() if grade == 2]
        assert answer_docs, qid
        assert any(ds.answers[qid][0] in text for text in answer_docs), qid
