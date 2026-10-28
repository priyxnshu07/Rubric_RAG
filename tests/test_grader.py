import json

import pytest

from app import grader
from app.schemas import Chunk

RUBRIC = [
    Chunk(id="R3", text="Q1 Inputs (3 points)", doc_type="rubric", start_line=3, end_line=3),
    Chunk(id="R4", text="Q1 Outputs (2 points)", doc_type="rubric", start_line=4, end_line=4),
]


class FakeStore:
    def __init__(self, rubric=RUBRIC):
        self.rubric = rubric

    def retrieve(self, rubric_id, query, doc_type, k):
        return self.rubric if doc_type == "rubric" else []


def fake_llm(payload):
    return lambda system, user: json.dumps(payload)


GOOD = {"criteria": [
    {"criterion": "Inputs", "score": 2, "max_score": 3, "feedback": "Missing CO2.", "citations": ["R3"]},
    {"criterion": "Outputs", "score": 2, "max_score": 2, "feedback": "Good.", "citations": ["R4"]},
]}


def test_grade_sums_total_in_code_and_resolves_citations():
    res = grader.grade(FakeStore(), "quiz", "some answer", llm=fake_llm(GOOD))
    assert res.total_score == 4 and res.max_score == 5
    assert [c.id for c in res.citations] == ["R3", "R4"]
    assert res.citations[0].text == "Q1 Inputs (3 points)"
    assert res.warnings == []


def test_hallucinated_citations_are_removed_and_flagged():
    bad = {"criteria": [{"criterion": "Inputs", "score": 3, "max_score": 3, "feedback": "ok",
                         "citations": ["R3", "R99"]}]}
    res = grader.grade(FakeStore(), "quiz", "answer", llm=fake_llm(bad))
    assert res.criteria[0].citations == ["R3"]
    assert any("R99" in w for w in res.warnings)


def test_criterion_without_valid_citation_is_flagged():
    bad = {"criteria": [{"criterion": "Inputs", "score": 3, "max_score": 3, "feedback": "ok", "citations": []}]}
    res = grader.grade(FakeStore(), "quiz", "answer", llm=fake_llm(bad))
    assert any("no valid rubric citation" in w for w in res.warnings)


def test_score_above_max_is_clamped():
    bad = {"criteria": [{"criterion": "Inputs", "score": 5, "max_score": 3, "feedback": "ok", "citations": ["R3"]}]}
    res = grader.grade(FakeStore(), "quiz", "answer", llm=fake_llm(bad))
    assert res.criteria[0].score == 3 and res.total_score == 3
    assert any("clamped" in w for w in res.warnings)


def test_malformed_llm_output_is_retried_once():
    replies = iter(["not json", json.dumps(GOOD)])
    res = grader.grade(FakeStore(), "quiz", "answer", llm=lambda s, u: next(replies))
    assert res.total_score == 4


def test_malformed_output_twice_raises():
    with pytest.raises(ValueError):
        grader.grade(FakeStore(), "quiz", "answer", llm=lambda s, u: "still not json")


def test_unknown_rubric_raises_lookup_error():
    with pytest.raises(LookupError):
        grader.grade(FakeStore(rubric=[]), "missing", "answer", llm=fake_llm(GOOD))


def test_provider_setting_picks_the_llm(monkeypatch):
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "gemini")
    assert grader.default_llm() is grader.gemini_llm
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "groq")
    assert grader.default_llm() is grader.groq_llm
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "openai")
    with pytest.raises(RuntimeError):
        grader.default_llm()


def test_missing_gemini_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(grader.config, "GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        grader.gemini_llm("system", "user")


def test_prompt_wraps_answer_and_lists_chunk_ids():
    prompt = grader.build_user_prompt("Explain photosynthesis", "my answer", RUBRIC, [])
    assert "[R3] Q1 Inputs (3 points)" in prompt
    assert "<answer>\nmy answer\n</answer>" in prompt
