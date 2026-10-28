"""Ingest a rubric folder into ChromaDB from the command line.

    python -m app.ingest data/biology-quiz-1

The folder must contain exactly one file with "rubric" in its name and any
number of files with "notes" in their name (.md, .txt or .pdf). Other files
are skipped. The folder name becomes the rubric id unless --rubric-id is given.
"""
import argparse
import sys
from pathlib import Path

from app import config
from app.loaders import SUPPORTED, read_file


def load_folder(folder: Path) -> tuple[tuple[str, str], list[tuple[str, str]], list[str]]:
    """Return (rubric (name, text), notes [(name, text)], skipped file names)."""
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a folder.")
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))
    rubrics = [p for p in files if "rubric" in p.name.lower() and p.suffix.lower() in SUPPORTED]
    notes = [p for p in files if "notes" in p.name.lower() and p.suffix.lower() in SUPPORTED]
    skipped = [p.name for p in files if p not in rubrics and p not in notes]
    if len(rubrics) != 1:
        found = ", ".join(p.name for p in rubrics) or "none"
        raise ValueError(f"Expected exactly one rubric file (name containing 'rubric') in {folder}, found: {found}")
    return (rubrics[0].name, read_file(rubrics[0])), [(p.name, read_file(p)) for p in notes], skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.ingest", description="Ingest a rubric folder into ChromaDB.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--rubric-id", help="defaults to the folder name")
    ap.add_argument("--chroma-dir", default=config.CHROMA_DIR)
    args = ap.parse_args(argv)

    try:
        rubric, notes, skipped = load_folder(args.folder)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    from app.store import Store  # imported here so --help and folder errors don't wait for the embedding model

    rubric_id = args.rubric_id or args.folder.name
    store = Store(path=args.chroma_dir)
    n_rubric, n_notes = store.ingest_documents(rubric_id, rubric, notes)

    print(f"Ingested '{rubric_id}' into {args.chroma_dir}: {n_rubric} rubric chunks, {n_notes} notes chunks")
    for c in store.get_chunks(rubric_id, "rubric"):
        label = c.criterion or "(general text)"
        pts = f" [{c.points:g} pts]" if c.points is not None else ""
        print(f"  {c.id:<8} {label}{pts}")
    if skipped:
        print(f"Skipped (not a rubric/notes file): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
