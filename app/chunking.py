"""Split rubric and notes text into chunks that remember where they came from.

Rubrics are split into one chunk per CRITERION, never mid-criterion. A
criterion starts at any line that states its points, e.g.

    ### Inputs (3 points)          <- markdown style, levels as bullets below
    Q1 Inputs (3 points): 3 = ...  <- one-line style

and runs until the next criterion or the next heading, so its level
descriptors (bullets, blank lines in between) stay in the same chunk. A
heading without points (e.g. "## Q1 Photosynthesis") is a section: it is not
a chunk itself, but it is prefixed to the criteria under it so retrieval
knows which question they belong to. Any other text (titles, instructions)
becomes a "general" chunk, split at blank lines.

Notes are split into paragraphs (blocks separated by blank lines). A heading
on its own is merged into the paragraph that follows it.
"""
import re

from app.schemas import Chunk

POINTS = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*(?:points?|pts?|marks?)\s*\)", re.IGNORECASE)
HEADING = re.compile(r"^#{1,6}\s+(.*)$")


def _clean(text: str) -> str:
    """Strip markdown markers and trailing separators from a criterion name."""
    return re.sub(r"^[#*\-\s]+", "", text).strip().rstrip(" -:–").replace("**", "")


def _chunk_id(prefix: str, start: int, end: int) -> str:
    return f"{prefix}{start}" if start == end else f"{prefix}{start}-{end}"


def chunk_rubric(text: str, source: str = "rubric") -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    block: dict | None = None  # the chunk being built

    def flush():
        nonlocal block
        if block and block["lines"]:
            body = "\n".join(block["lines"])
            chunks.append(Chunk(
                id=_chunk_id("R", block["start"], block["end"]),
                text=f"{section} > {body}" if section else body,
                doc_type="rubric",
                source=source,
                criterion=block["criterion"],
                points=block["points"],
                start_line=block["start"],
                end_line=block["end"],
            ))
        block = None

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        points = POINTS.search(line)
        heading = HEADING.match(line)

        if points:  # a new criterion starts here
            flush()
            name = _clean(line[:points.start()])
            block = {"start": line_no, "end": line_no, "lines": [re.sub(r"^#{1,6}\s+", "", line)],
                     "points": float(points.group(1)),
                     "criterion": f"{section} > {name}" if section else name}
        elif heading:  # a section heading: closes the current chunk, labels the next ones
            flush()
            section = _clean(heading.group(1))
        elif line:
            if block is None:
                block = {"start": line_no, "end": line_no, "lines": [], "points": None, "criterion": ""}
            block["lines"].append(line)
            block["end"] = line_no
        elif block and block["criterion"] == "":
            flush()  # blank line ends a general paragraph, but NOT a criterion

    flush()
    return chunks


def chunk_notes(text: str, source: str = "notes", prefix: str = "N") -> list[Chunk]:
    chunks = []
    block: list[str] = []
    start = 0
    pending_heading = ""
    for line_no, line in enumerate(text.splitlines() + [""], start=1):  # trailing "" flushes the last block
        if line.strip():
            if not block:
                start = line_no
            block.append(line.strip())
        elif block:
            end = line_no - 1
            heading = HEADING.match(block[0])
            if len(block) == 1 and heading:  # lone heading: attach it to the next paragraph
                pending_heading = heading.group(1).strip()
            else:
                body = " ".join(re.sub(r"^#{1,6}\s+", "", b) for b in block)
                chunks.append(Chunk(
                    id=_chunk_id(prefix, start, end),
                    text=f"{pending_heading}: {body}" if pending_heading else body,
                    doc_type="notes", source=source, start_line=start, end_line=end,
                ))
                pending_heading = ""
            block = []
    return chunks
