import pytest

from rag_eval.datasets import Dataset, Document
from rag_eval.generation import Completion, ExtractiveGenerator, LatencyModel, LLMGenerator
from rag_eval.generation.llm import parse_search_reply
from rag_eval.generation.mock import extract_span
from rag_eval.generation.prompts import ABSTAIN
from rag_eval.judge import HeuristicJudge, LLMJudge, Verdict, parse_verdict
from rag_eval.text import content_terms
from tests.fakes import ScriptedClient


def test_extract_span_prefers_copula_predicate() -> None:
    terms = frozenset(content_terms("What is the Xero sync interval?"))
    assert extract_span("The Xero sync interval is every 15 minutes.", terms) == "every 15 minutes"


def test_extract_span_falls_back_to_novel_window() -> None:
    terms = frozenset(content_terms("When does Slack deliver alerts?"))
    sentence = "Slack alerts arrive within 1 minute of the event."
    assert extract_span(sentence, terms) == "arrive within 1 minute of the event"


def test_extractive_answer_cites_source(tiny_dataset: Dataset) -> None:
    gen = ExtractiveGenerator(price_as="m")
    contexts = [tiny_dataset.corpus["d2"], tiny_dataset.corpus["d1"]]
    out = gen.answer("What is the API rate limit on the Team plan?", contexts)
    assert out.text == "300 requests [d1]"
    assert out.latency_modeled
    (usage,) = out.usage
    assert usage.model == "m"
    assert usage.input_tokens > 50
    assert out.latency_ms == pytest.approx(
        LatencyModel().estimate(usage.input_tokens, usage.output_tokens)
    )


def test_extractive_abstains_without_support(tiny_dataset: Dataset) -> None:
    gen = ExtractiveGenerator()
    out = gen.answer("Who founded the company?", [tiny_dataset.corpus["d4"]])
    assert out.text == ABSTAIN
    assert gen.answer("anything", []).text == ABSTAIN


def test_planner_substitutes_bridge_fact() -> None:
    contexts = [
        Document("rl-biz", "", "The API rate limit on the Business plan is 1200."),
        Document("ri", "", "The minimum plan for recurring invoices is Team."),
    ]
    gen = ExtractiveGenerator()
    question = "What is the API rate limit on the cheapest plan that includes recurring invoices?"
    step = gen.next_search(question, contexts, [])
    assert step.text == "api rate limit cheapest include minimum team"
    # Once the hop has been run, the same query is not proposed again.
    assert gen.next_search(question, contexts, [step.text]).text == ""


def test_planner_stops_when_covered(tiny_dataset: Dataset) -> None:
    gen = ExtractiveGenerator()
    question = "What is the API rate limit on the Team plan?"
    assert gen.next_search(question, [tiny_dataset.corpus["d1"]], []).text == ""


def test_answer_targets_latest_search() -> None:
    contexts = [
        Document("ri", "", "The minimum plan for recurring invoices is Team."),
        Document("rl-team", "", "The API rate limit on the Team plan is 300."),
    ]
    gen = ExtractiveGenerator()
    out = gen.answer("question", contexts, ["api rate limit team"])
    assert out.text == "300 [rl-team]"


def test_completion_round_trip() -> None:
    gen = ExtractiveGenerator()
    out = gen.answer("What formats?", [Document("d", "", "Formats are CSV and JSON.")])
    assert Completion.from_dict(out.to_dict()) == out


def test_llm_generator_uses_client_usage_and_parses_search() -> None:
    client = ScriptedClient(lambda s, u: "SEARCH: business plan rate limit\nextra")
    gen = LLMGenerator(client, price_as="claude-haiku-4-5")
    step = gen.next_search("q", [Document("d", "T", "text")], [])
    assert step.text == "business plan rate limit"
    assert step.usage[0].model == "claude-haiku-4-5"
    assert step.usage[0].output_tokens == 7
    assert not step.latency_modeled
    assert "[d] T: text" in client.calls[0][1]
    assert gen.name == "llm(fake:fake-model,max_tokens=256)"
    assert gen.answer("q", []).text.startswith("SEARCH")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [("DONE", ""), ("done.", ""), ("", ""), ("SEARCH:", ""), ("plain query", "plain query")],
)
def test_parse_search_reply(reply: str, expected: str) -> None:
    assert parse_search_reply(reply) == expected


def test_heuristic_judge_rubric() -> None:
    docs = [Document("d1", "", "The Xero sync interval is every 15 minutes.")]
    judge = HeuristicJudge()
    assert judge.judge("q", "every 15 minutes [d1]", ["every 15 minutes"], docs).score == 5
    assert judge.judge("q", "every 15 minutes", ["every 15 minutes"], docs).score == 4  # uncited
    assert judge.judge("q", ABSTAIN, ["every 15 minutes"], docs).score == 1
    assert judge.judge("q", "every 15 minutes [d1]", [], docs).score == 3
    assert Verdict(5, "").normalized == 1.0
    assert Verdict(1, "").normalized == 0.0


def test_llm_judge_parses_score() -> None:
    client = ScriptedClient(lambda s, u: "Score: 4\nCorrect but verbose.")
    verdict = LLMJudge(client).judge("q", "a", ["a"], [])
    assert (verdict.score, verdict.rationale) == (4, "Correct but verbose")
    assert Verdict.from_dict(verdict.to_dict()) == verdict
    assert "REFERENCE ANSWERS: a" in client.calls[0][1]
    with pytest.raises(ValueError):
        parse_verdict("looks fine")
