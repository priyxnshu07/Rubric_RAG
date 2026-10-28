"""End-to-end tests with a real ChromaDB store (in a temp folder) and a fake LLM."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.store import Store

DATA = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = Store(path=str(tmp_path_factory.mktemp("chroma")))
    s.ingest("biology-quiz-1", (DATA / "biology_quiz_rubric.txt").read_text(),
             (DATA / "biology_notes.txt").read_text())
    return s


@pytest.fixture
def client(store):
    def fake_llm(system, user):
        return json.dumps({"criteria": [{"criterion": "Location", "score": 2, "max_score": 2,
                                         "feedback": "Correct.", "citations": ["R9"]}]})

    main.app.dependency_overrides[main.get_store] = lambda: store
    main.app.dependency_overrides[main.get_llm] = lambda: fake_llm
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_retrieval_finds_the_right_questions_criteria(store):
    chunks = store.retrieve("biology-quiz-1", "Where does cellular respiration happen? In the mitochondria.",
                            "rubric", k=6)
    ids = {c.id for c in chunks}
    assert {"R8", "R9", "R10", "R11"} <= ids  # all four Q2 criteria


def test_reingest_replaces_old_chunks(tmp_path):
    s = Store(path=str(tmp_path))
    s.ingest("r", "line a\nline b\nline c")
    s.ingest("r", "only line")
    assert [c.id for c in s.retrieve("r", "line", "rubric", k=10)] == ["R1"]


def test_list_rubrics(client):
    assert client.get("/rubrics").json() == ["biology-quiz-1"]


def test_grade_endpoint(client):
    res = client.post("/grade", json={"rubric_id": "biology-quiz-1", "answer": "It happens in the mitochondria."})
    assert res.status_code == 200
    body = res.json()
    assert body["total_score"] == 2
    assert body["citations"][0]["id"] == "R9"
    assert "mitochondria" in body["citations"][0]["text"]


def test_grade_unknown_rubric_returns_404(client):
    res = client.post("/grade", json={"rubric_id": "nope", "answer": "x"})
    assert res.status_code == 404


def test_ingest_files_endpoint(tmp_path):
    main.app.dependency_overrides[main.get_store] = lambda: Store(path=str(tmp_path))
    try:
        res = TestClient(main.app).post(
            "/ingest/files", data={"rubric_id": "t"},
            files={"rubric": ("r.txt", b"Q1 Clarity (2 points)\n"), "notes": ("n.txt", b"para one\n\npara two")},
        )
    finally:
        main.app.dependency_overrides.clear()
    assert res.status_code == 200
    assert res.json() == {"rubric_id": "t", "rubric_chunks": 1, "notes_chunks": 2}
