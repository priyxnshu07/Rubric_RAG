"""End-to-end tests with a real ChromaDB store (in a temp folder) and a fake LLM."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import grader, main
from app.ingest import load_folder
from app.store import Store

SAMPLE = Path(__file__).parent.parent / "data" / "biology-quiz-1"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(path=str(tmp_path_factory.mktemp("chroma")))
    rubric, notes, _ = load_folder(SAMPLE)
    s.ingest_documents("biology-quiz-1", rubric, notes)
    return s


def chunk_id(store, criterion):
    return next(c.id for c in store.get_chunks("biology-quiz-1", "rubric") if c.criterion == criterion)


GRADE_REQUEST = {"rubric_collection": "biology-quiz-1", "question": "Where does respiration happen?",
                 "answer": "It happens in the mitochondria."}


@pytest.fixture
def llm_reply(store):
    """Mutable holder for what the mocked LLM returns; tests can swap in a function."""
    location = chunk_id(store, "Q2 Cellular respiration > Location")
    return {"fn": lambda messages: json.dumps({
        "score": 2, "max_score": 2,
        "criteria": [{"name": "Location", "met": True, "feedback": "Correct.", "cited_chunk_ids": [location]}],
        "overall_feedback": "Correct location.", "needs_human_review": False, "reasons": []})}


@pytest.fixture
def client(store, llm_reply):
    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_llm] = lambda: (lambda messages: llm_reply["fn"](messages))
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_retrieval_finds_the_right_questions_criteria(store):
    chunks = store.retrieve("biology-quiz-1", "Where does cellular respiration happen? In the mitochondria.",
                            "rubric", k=6)
    criteria = {c.criterion for c in chunks}
    assert {f"Q2 Cellular respiration > {n}" for n in ["Equation", "Location", "ATP", "Aerobic vs anaerobic"]} <= criteria


def test_reingest_replaces_old_chunks(tmp_path):
    s = Store(path=str(tmp_path))
    s.ingest("r", "line a\nline b\nline c")
    s.ingest("r", "only line")
    assert [c.id for c in s.retrieve("r", "line", "rubric", k=10)] == ["R1"]


def test_list_rubrics(client):
    assert client.get("/rubrics").json() == ["biology-quiz-1"]


def test_grade_endpoint_returns_exact_schema(client, store):
    res = client.post("/grade", json=GRADE_REQUEST)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"score", "max_score", "criteria", "overall_feedback", "needs_human_review", "reasons"}
    assert set(body["criteria"][0]) == {"name", "met", "feedback", "cited_chunk_ids"}
    assert body["score"] == 2 and body["needs_human_review"] is False
    assert body["criteria"][0]["cited_chunk_ids"] == [chunk_id(store, "Q2 Cellular respiration > Location")]


def test_off_topic_answer_is_flagged_as_weak_match_with_real_embeddings(client):
    res = client.post("/grade", json={**GRADE_REQUEST, "answer": "My favourite football team won yesterday."})
    body = res.json()
    assert body["needs_human_review"] and "weak rubric match" in body["reasons"]


def test_retrieval_returns_similarity_scores(store):
    hits = store.retrieve("biology-quiz-1", "It happens in the mitochondria", "rubric", k=4)
    sims = [c.similarity for c in hits]
    assert all(s is not None and -1 <= s <= 1 for s in sims)


def test_chunks_endpoint_resolves_cited_ids(client, store):
    chunks = {c["id"]: c for c in client.get("/rubrics/biology-quiz-1/chunks").json()}
    loc = chunks[chunk_id(store, "Q2 Cellular respiration > Location")]
    assert "mitochondria" in loc["text"] and loc["source"] == "rubric.md"


def test_grade_unknown_collection_returns_404(client):
    res = client.post("/grade", json={**GRADE_REQUEST, "rubric_collection": "nope"})
    assert res.status_code == 404 and "'nope' not found" in res.json()["detail"]


@pytest.mark.parametrize("missing", ["question", "answer", "rubric_collection"])
def test_grade_missing_field_returns_422(client, missing):
    body = {k: v for k, v in GRADE_REQUEST.items() if k != missing}
    res = client.post("/grade", json=body)
    assert res.status_code == 422 and missing in str(res.json()["detail"])


def test_grade_invalid_llm_output_returns_502_with_reason(client, llm_reply):
    llm_reply["fn"] = lambda messages: "this is not json"
    res = client.post("/grade", json=GRADE_REQUEST)
    assert res.status_code == 502
    assert "invalid output 3 times" in res.json()["detail"]


def test_grade_provider_failure_returns_502(client, llm_reply):
    def down(messages):
        raise grader.LLMProviderError("Groq API error: 503 service unavailable")
    llm_reply["fn"] = down
    res = client.post("/grade", json=GRADE_REQUEST)
    assert res.status_code == 502 and "LLM provider error" in res.json()["detail"]


def test_grade_missing_key_returns_500(client, llm_reply):
    def no_key(messages):
        raise grader.LLMConfigError("GROQ_API_KEY is not set.")
    llm_reply["fn"] = no_key
    res = client.post("/grade", json=GRADE_REQUEST)
    assert res.status_code == 500 and "GROQ_API_KEY" in res.json()["detail"]


def test_ingest_files_endpoint(tmp_path):
    main.app.dependency_overrides[main.get_store] = lambda: Store(path=str(tmp_path))
    try:
        res = TestClient(main.app).post(
            "/ingest/files", data={"rubric_id": "t"},
            files={"rubric": ("r.md", b"Q1 Clarity (2 points)\n"), "notes": ("n.txt", b"para one\n\npara two")},
        )
    finally:
        main.app.dependency_overrides.clear()
    assert res.status_code == 200
    assert res.json() == {"rubric_id": "t", "rubric_chunks": 1, "notes_chunks": 2}


def test_ingest_files_rejects_unsupported_type(tmp_path):
    main.app.dependency_overrides[main.get_store] = lambda: Store(path=str(tmp_path))
    try:
        res = TestClient(main.app).post("/ingest/files", data={"rubric_id": "t"},
                                        files={"rubric": ("r.docx", b"binary")})
    finally:
        main.app.dependency_overrides.clear()
    assert res.status_code == 400 and "Unsupported file type" in res.json()["detail"]
