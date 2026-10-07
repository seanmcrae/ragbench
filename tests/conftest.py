from __future__ import annotations

import pytest

from rag_eval.datasets import Dataset, Document, Query


@pytest.fixture
def tiny_dataset() -> Dataset:
    """Hand-written five-document fixture for retriever and runner tests."""
    corpus = {
        "d1": Document("d1", "Rate limits", "The API rate limit on the Team plan is 300 requests."),
        "d2": Document("d2", "Rate limits", "The API rate limit on the Business plan is 1200."),
        "d3": Document("d3", "Recurring invoices", "Recurring invoices are available on Team."),
        "d4": Document("d4", "Exports", "Data exports are available as CSV and JSON files."),
        "d5": Document("d5", "Slack", "The Slack integration is synced every 5 minutes."),
    }
    queries = {
        "q1": Query("q1", "What is the API rate limit on the Team plan?"),
        "q2": Query("q2", "Which formats can data exports use?"),
        "q3": Query("q3", "How often does Slack sync?"),
    }
    qrels = {"q1": {"d1": 2, "d2": 1}, "q2": {"d4": 2}, "q3": {"d5": 2}}
    answers = {"q1": ["300 requests"], "q2": ["CSV and JSON files"], "q3": ["every 5 minutes"]}
    return Dataset("tiny", corpus, queries, qrels, answers)
