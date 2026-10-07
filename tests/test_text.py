from rag_eval.text import (
    content_terms,
    count_tokens,
    normalize_answer,
    split_sentences,
    tokenize,
)


def test_tokenize_keeps_codes_and_hyphenated_words() -> None:
    assert tokenize("Error TH-4031: re-connect Slack!") == ["error", "th-4031", "re-connect", "slack"]


def test_content_terms_drop_stopwords_and_fold_plurals() -> None:
    assert content_terms("What are the invoices for this plan?") == ["invoice", "plan"]
    assert content_terms("access status") == ["access", "status"]


def test_split_sentences() -> None:
    text = "First sentence. Second one? 3 more. trailing"
    assert split_sentences(text) == ["First sentence.", "Second one?", "3 more. trailing"]


def test_normalize_answer_squad_style() -> None:
    assert normalize_answer("The  Team plan, and above.") == "team plan and above"


def test_count_tokens_approximation() -> None:
    assert count_tokens("") == 0
    assert count_tokens("hello") == 2
    assert count_tokens("a, b.") == 4
