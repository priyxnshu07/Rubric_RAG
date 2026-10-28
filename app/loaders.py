"""Turn uploaded or on-disk files (.md, .txt, .pdf) into plain text."""
import io
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf"}


def read_document(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}' ({filename}). Use .md, .txt or .pdf.")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError(f"No text found in {filename}. Scanned PDFs (images) are not supported.")
        return text
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{filename} is not UTF-8 text.")


def read_file(path: Path) -> str:
    return read_document(path.name, path.read_bytes())
