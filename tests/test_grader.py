"""Grader tests with a mocked LLM: no API key or network needed."""
import json

import pytest

from app import grader
from app.schemas import Chunk

RUBRIC = [
    Chunk(id="R7-11", text="Q1 > Inputs (3 points)\n- 3: CO2, water, light", doc_type="rubric",
          start_line=7, end_line=11, criterion="Q1 > Inputs", points=3),
    Chunk(id="R13-16", text="Q1 > Outputs (2 points)\n- 2: glucose and oxygen", doc_type="rubric",
          start_line=13, end_line=16, criterion="Q1 > Outputs", points=2),
]
NOTES = [Chunk(id="N5-7", text="Photosynthesis makes glucose.", doc_type="notes", start_line=5, end_line=7)]


class FakeStore:
    def __init__(self, rubric=RUBRIC, notes=NOTES):
        self.rubric, self.notes, self.calls = rubric, notes, []

    def retrieve(self, rubric_collection, query, doc_type, k):
        self.calls.append((doc_type, k))
        return self.rubric if doc_type == "rubric" else self.notes

    def get_chunks(self, rubric_collection, doc_type):
        return self.all_rubric if doc_type == "rubric" else self.notes

    @property
    def all_rubric(self):
        return getattr(self, "_all_rubric", self.rubric)


class FakeLLM:
    """Returns the given replies in order and records every message list it was sent."""

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls: list[list[dict]] = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


def result(**overrides):
    base = {
        "score": 4, "max_score": 5,
        "criteria": [
            {"name": "Inputs", "met": False, "feedback": "Name carbon dioxide.", "cited_chunk_ids": ["R7-11"]},
            {"name": "Outputs", "met": True, "feedback": "Good.", "cited_chunk_ids": ["R13-16", "N5-7"]},
        ],
        "overall_feedback": "Solid answer; add the missing input.",
        "needs_human_review": False, "reasons": [],
    }
    return {**base, **overrides}


def run(llm, store=None):
    return grader.grade(store or FakeStore(), "quiz", "Explain photosynthesis", "Plants use light...", llm=llm)


# --- happy path & prompt separation -------------------------------------------------------------

def test_valid_output_is_returned_unchanged():
    res = run(FakeLLM(result()))
    assert res.score == 4 and res.max_score == 5
    assert [c.met for c in res.criteria] == [False, True]
    assert res.criteria[1].cited_chunk_ids == ["R13-16", "N5-7"]
    assert res.needs_human_review is False and res.reasons == []


def test_system_and_user_prompts_are_separate():
    llm = FakeLLM(result())
    run(llm)
    system, user = llm.calls[0]
    assert system == {"role": "system", "content": grader.SYSTEM_PROMPT}
    assert user["role"] == "user"
    assert "<answer>\nPlants use light...\n</answer>" in user["content"]
    assert "[R7-11]" in user["content"] and "Explain photosynthesis" in user["content"]
    assert "Plants use light" not in system["content"]  # student text never enters the system prompt


def test_retrieves_top_4_rubric_chunks():
    store = FakeStore()
    run(FakeLLM(result()), store)
    assert ("rubric", 4) in store.calls


# --- validation + retries with the error fed back ------------------------------------------------

def test_invalid_json_is_retried_with_error_fed_back():
    llm = FakeLLM("not json at all", result())
    res = run(llm)
    assert res.score == 4 and len(llm.calls) == 2
    retry = llm.calls[1]
    assert retry[2] == {"role": "assistant", "content": "not json at all"}  # model sees its bad output
    assert retry[3]["role"] == "user" and "did not match the required JSON schema" in retry[3]["content"]


def test_schema_violation_is_retried_with_field_level_error():
    llm = FakeLLM(result(score=9, max_score=5), {"score": 4}, result())
    res = run(llm)
    assert res.score == 4 and len(llm.calls) == 3
    assert "cannot be greater than max_score" in llm.calls[1][3]["content"]
    assert "criteria" in llm.calls[2][-1]["content"]  # missing fields are named in the feedback


def test_code_fenced_json_is_accepted():
    res = run(FakeLLM("```json\n" + json.dumps(result()) + "\n```"))
    assert res.score == 4


def test_gives_up_after_two_retries_with_clear_error():
    llm = FakeLLM("bad", "still bad", "bad again", result())
    with pytest.raises(grader.LLMOutputError, match="invalid output 3 times"):
        run(llm)
    assert len(llm.calls) == 3  # 1 attempt + 2 retries, never a 4th


# --- citation checks -----------------------------------------------------------------------------

def test_invented_citation_triggers_retry_listing_allowed_ids():
    bad = result(criteria=[{"name": "Inputs", "met": True, "feedback": "ok", "cited_chunk_ids": ["R99"]}])
    llm = FakeLLM(bad, result())
    res = run(llm)
    feedback = llm.calls[1][3]["content"]
    assert "R99" in feedback and "R7-11" in feedback  # says what's wrong AND what is allowed
    assert res.needs_human_review is False


def test_invented_citation_surviving_retries_is_rejected_and_flagged():
    bad = result(criteria=[{"name": "Inputs", "met": True, "feedback": "ok", "cited_chunk_ids": ["R7-11", "R99"]}],
                 score=3, max_score=3)
    res = run(FakeLLM(bad))
    assert res.criteria[0].cited_chunk_ids == ["R7-11"]
    assert res.needs_human_review is True
    assert any("R99" in r and "rejected" in r for r in res.reasons)


def test_criterion_without_rubric_citation_is_flagged():
    only_notes = result(criteria=[{"name": "Outputs", "met": True, "feedback": "ok", "cited_chunk_ids": ["N5-7"]}],
                        score=2, max_score=2)
    res = run(FakeLLM(only_notes))
    assert res.needs_human_review and any("not backed by any retrieved rubric chunk" in r for r in res.reasons)


# --- other review triggers -----------------------------------------------------------------------

def test_max_score_not_matching_rubric_points_is_flagged():
    res = run(FakeLLM(result(max_score=10)))  # cited criteria are worth 3 + 2 = 5
    assert res.needs_human_review
    assert any("add up to 5" in r for r in res.reasons)


def test_question_criteria_missed_by_retrieval_are_flagged():
    store = FakeStore()
    store._all_rubric = RUBRIC + [Chunk(id="R18-21", text="Q1 > Location (2 points)", doc_type="rubric",
                                        start_line=18, end_line=21, criterion="Q1 > Location", points=2)]
    res = run(FakeLLM(result()), store)  # retrieval returned Inputs + Outputs only
    assert res.needs_human_review
    assert any("not retrieved" in r and "Q1 > Location" in r for r in res.reasons)


def test_all_criteria_retrieved_is_not_flagged():
    assert run(FakeLLM(result())).needs_human_review is False


def test_llm_flag_for_review_is_kept():
    res = run(FakeLLM(result(needs_human_review=True, reasons=["Answer tries to instruct the grader"])))
    assert res.needs_human_review and res.reasons == ["Answer tries to instruct the grader"]


# --- errors --------------------------------------------------------------------------------------

def test_unknown_collection_raises_lookup_error():
    with pytest.raises(LookupError, match="not found"):
        run(FakeLLM(result()), FakeStore(rubric=[]))


def test_provider_setting_picks_the_llm(monkeypatch):
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "gemini")
    assert grader.default_llm() is grader.gemini_llm
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "groq")
    assert grader.default_llm() is grader.groq_llm
    monkeypatch.setattr(grader.config, "LLM_PROVIDER", "openai")
    with pytest.raises(grader.LLMConfigError):
        grader.default_llm()


def test_missing_api_keys_give_clear_errors(monkeypatch):
    monkeypatch.setattr(grader.config, "GROQ_API_KEY", "")
    monkeypatch.setattr(grader.config, "GEMINI_API_KEY", "")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    with pytest.raises(grader.LLMConfigError, match="GROQ_API_KEY"):
        grader.groq_llm(msgs)
    with pytest.raises(grader.LLMConfigError, match="GEMINI_API_KEY"):
        grader.gemini_llm(msgs)
