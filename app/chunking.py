"""Split rubric and notes text into chunks that remember their line numbers.

Rubrics are split one chunk per non-empty line, because each rubric line is
usually a self-contained rule ("2 pts: names both reactants") and citing a
single line is the most useful feedback for a student.

Notes are split into paragraphs (blocks separated by blank lines), since a
single line of prose rarely makes sense on its own.
"""
from app.schemas import Chunk


def chunk_rubric(text: str) -> list[Chunk]:
    chunks = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if line:
            chunks.append(Chunk(id=f"R{line_no}", text=line, doc_type="rubric",
                                start_line=line_no, end_line=line_no))
    return chunks


def chunk_notes(text: str) -> list[Chunk]:
    chunks = []
    block: list[str] = []
    start = 0
    lines = text.splitlines()
    for line_no, line in enumerate(lines + [""], start=1):  # trailing "" flushes the last block
        if line.strip():
            if not block:
                start = line_no
            block.append(line.strip())
        elif block:
            end = line_no - 1
            chunks.append(Chunk(id=f"N{start}-{end}", text=" ".join(block), doc_type="notes",
                                start_line=start, end_line=end))
            block = []
    return chunks
