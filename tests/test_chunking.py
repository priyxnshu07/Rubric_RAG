from app.chunking import chunk_notes, chunk_rubric

MARKDOWN_RUBRIC = """# Essay rubric

Read the whole essay before marking.

## Q1 Argument

### Thesis (4 points)
- 4: clear, arguable thesis

- 2: thesis is vague
- 0: no thesis

### Evidence (6 points)
- 6: three relevant sources
- 0: none
"""


def test_markdown_criterion_stays_in_one_chunk_even_across_blank_lines():
    chunks = chunk_rubric(MARKDOWN_RUBRIC, source="rubric.md")
    thesis = next(c for c in chunks if c.criterion.endswith("Thesis"))
    assert "4: clear, arguable thesis" in thesis.text and "0: no thesis" in thesis.text
    assert "three relevant sources" not in thesis.text  # the next criterion is NOT merged in
    assert (thesis.start_line, thesis.end_line) == (7, 11) and thesis.id == "R7-11"


def test_criterion_metadata_has_section_name_points_and_source():
    criteria = [c for c in chunk_rubric(MARKDOWN_RUBRIC, source="rubric.md") if c.criterion]
    assert [c.criterion for c in criteria] == ["Q1 Argument > Thesis", "Q1 Argument > Evidence"]
    assert [c.points for c in criteria] == [4, 6]
    assert all(c.source == "rubric.md" for c in criteria)
    assert criteria[0].text.startswith("Q1 Argument > ")  # section prefixed so retrieval knows the question


def test_text_outside_criteria_becomes_general_chunk():
    general = [c for c in chunk_rubric(MARKDOWN_RUBRIC) if not c.criterion]
    assert len(general) == 1 and "Read the whole essay" in general[0].text
    assert general[0].points is None


def test_one_line_criteria_format_still_works():
    chunks = chunk_rubric("Title\n\n  Q1 Inputs (3 points): 3 = all  \nQ1 Outputs (2 pts): 2 = both\n")
    assert [c.id for c in chunks] == ["R1", "R3", "R4"]
    assert chunks[1].criterion == "Q1 Inputs" and chunks[1].points == 3
    assert chunks[2].criterion == "Q1 Outputs" and chunks[2].points == 2


def test_notes_split_into_paragraphs_with_line_ranges():
    chunks = chunk_notes("line one\nline two\n\n\nline five\n")
    assert [c.id for c in chunks] == ["N1-2", "N5"]
    assert chunks[0].text == "line one line two"
    assert all(c.doc_type == "notes" for c in chunks)


def test_notes_heading_is_merged_into_next_paragraph():
    chunks = chunk_notes("## Photosynthesis\n\nPlants make glucose.\n", source="notes.md")
    assert len(chunks) == 1
    assert chunks[0].text == "Photosynthesis: Plants make glucose."
    assert chunks[0].source == "notes.md"


def test_empty_text_gives_no_chunks():
    assert chunk_rubric("") == []
    assert chunk_notes("\n\n") == []
