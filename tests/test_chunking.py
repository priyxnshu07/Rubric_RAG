from app.chunking import chunk_notes, chunk_rubric


def test_rubric_one_chunk_per_nonempty_line_with_line_numbers():
    chunks = chunk_rubric("Title\n\n  Q1 Inputs (3 points)  \nQ1 Outputs (2 points)\n")
    assert [c.id for c in chunks] == ["R1", "R3", "R4"]
    assert chunks[1].text == "Q1 Inputs (3 points)"  # whitespace stripped
    assert chunks[1].start_line == chunks[1].end_line == 3


def test_notes_split_into_paragraphs_with_line_ranges():
    chunks = chunk_notes("line one\nline two\n\n\nline five\n")
    assert [c.id for c in chunks] == ["N1-2", "N5-5"]
    assert chunks[0].text == "line one line two"
    assert all(c.doc_type == "notes" for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_rubric("") == []
    assert chunk_notes("\n\n") == []
