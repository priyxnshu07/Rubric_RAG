"""Tests for the `python -m app.ingest <folder>` command and folder/PDF loading."""
from pathlib import Path

import pytest

from app.ingest import load_folder, main
from app.store import Store

ROOT = Path(__file__).parent.parent
SAMPLE = ROOT / "data" / "biology-quiz-1"
PDF_FOLDER = Path(__file__).parent / "fixtures" / "pdf-rubric"


@pytest.fixture(scope="module")
def ingested(tmp_path_factory):
    chroma = str(tmp_path_factory.mktemp("chroma"))
    assert main([str(SAMPLE), "--chroma-dir", chroma]) == 0
    return Store(path=chroma)


def test_cli_creates_one_chunk_per_criterion(ingested):
    rubric = ingested.get_chunks("biology-quiz-1", "rubric")
    criteria = [c for c in rubric if c.criterion]
    assert len(criteria) == 12  # 4 criteria x 3 questions
    assert sum(c.points for c in criteria) == 30  # 10 points per question: no criterion lost or split
    assert all(c.source == "rubric.md" for c in rubric)
    assert len(ingested.get_chunks("biology-quiz-1", "notes")) > 0


def test_chunks_are_retrievable_with_metadata(ingested):
    hits = ingested.retrieve("biology-quiz-1", "Respiration happens in the mitochondria", "rubric", k=3)
    top = {c.criterion for c in hits}
    assert "Q2 Cellular respiration > Location" in top
    loc = next(c for c in hits if c.criterion == "Q2 Cellular respiration > Location")
    assert "mitochondria" in loc.text and loc.points == 2 and loc.id.startswith("R")


def test_general_text_never_takes_a_criterion_slot(tmp_path):
    """Regression: the public speaking intro text used to fill one of the top-4 slots, dropping a criterion."""
    assert main([str(ROOT / "data" / "public_speaking"), "--chroma-dir", str(tmp_path)]) == 0
    s = Store(path=str(tmp_path))
    query = "Write a short speech script to persuade your class.\nDid you know libraries are great?"
    hits = s.retrieve("public_speaking", query, "rubric", k=4, criteria_only=True)
    assert len(hits) == 4 and all(c.criterion for c in hits)
    assert sum(c.points for c in hits) == 8
    # Without the filter, general text competes for the slots.
    assert any(not c.criterion for c in s.retrieve("public_speaking", query, "rubric", k=4))


def test_criteria_only_falls_back_when_rubric_has_no_criteria(tmp_path):
    s = Store(path=str(tmp_path))
    s.ingest("plain", "Be clear.\n\nBe kind.")
    assert len(s.retrieve("plain", "clear", "rubric", k=4, criteria_only=True)) == 2


def test_cli_rubric_id_flag(tmp_path):
    assert main([str(SAMPLE), "--chroma-dir", str(tmp_path), "--rubric-id", "custom"]) == 0
    assert Store(path=str(tmp_path)).list_rubrics() == ["custom"]


def test_cli_fails_cleanly_without_rubric_file(tmp_path, capsys):
    (tmp_path / "notes.md").write_text("some notes")
    assert main([str(tmp_path), "--chroma-dir", str(tmp_path / "db")]) == 1
    assert "Expected exactly one rubric file" in capsys.readouterr().err


def test_load_folder_skips_unrelated_files(tmp_path):
    (tmp_path / "rubric.md").write_text("Clarity (2 points)\n- 2: clear")
    (tmp_path / "notes.txt").write_text("notes")
    (tmp_path / "answers.md").write_text("not ingested")
    rubric, notes, skipped = load_folder(tmp_path)
    assert rubric[0] == "rubric.md" and [n[0] for n in notes] == ["notes.txt"] and skipped == ["answers.md"]


def test_pdf_rubric_is_loaded_and_chunked_by_criterion(tmp_path):
    assert main([str(PDF_FOLDER), "--chroma-dir", str(tmp_path)]) == 0
    criteria = [c for c in Store(path=str(tmp_path)).get_chunks("pdf-rubric", "rubric") if c.criterion]
    assert [(c.criterion, c.points) for c in criteria] == [("Thesis", 4), ("Evidence", 6)]
    assert "thesis is vague" in criteria[0].text and criteria[0].source == "rubric.pdf"


def test_multiple_notes_files_get_unique_ids(tmp_path):
    s = Store(path=str(tmp_path))
    s.ingest_documents("r", ("rubric.md", "Clarity (2 points)"), [("a_notes.md", "para"), ("b_notes.md", "para")])
    assert sorted(c.id for c in s.get_chunks("r", "notes")) == ["a_notes:N1", "b_notes:N1"]
