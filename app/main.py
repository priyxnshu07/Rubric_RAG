"""FastAPI backend. Run with: uvicorn app.main:app --reload"""
import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from app import grader
from app.loaders import read_document
from app.schemas import Chunk, GradeRequest, GradeResponse, IngestRequest, IngestResponse
from app.store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="RubricRAG", description="Rubric-grounded feedback assistant")


@lru_cache
def get_store() -> Store:
    return Store()  # created once; loading the embedding model is slow


def get_llm() -> grader.LLM:
    try:
        return grader.default_llm()
    except grader.LLMConfigError as e:
        raise HTTPException(500, f"Server configuration error: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rubrics", response_model=list[str])
def list_rubrics(store: Store = Depends(get_store)):
    return store.list_rubrics()


@app.get("/rubrics/{rubric_collection}/chunks", response_model=list[Chunk])
def rubric_chunks(rubric_collection: str, store: Store = Depends(get_store)):
    """All rubric chunks of a collection, so clients can show the text behind cited_chunk_ids."""
    chunks = store.get_chunks(rubric_collection, "rubric") + store.get_chunks(rubric_collection, "notes")
    if not chunks:
        raise HTTPException(404, f"Rubric collection '{rubric_collection}' not found.")
    return chunks


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, store: Store = Depends(get_store)):
    r, n = store.ingest(req.rubric_id, req.rubric_text, req.notes_text)
    return IngestResponse(rubric_id=req.rubric_id, rubric_chunks=r, notes_chunks=n)


@app.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    rubric_id: str = Form(...),
    rubric: UploadFile = File(...),
    notes: UploadFile | None = File(None),
    store: Store = Depends(get_store),
):
    """Same as /ingest, but takes .md / .txt / .pdf file uploads."""
    try:
        rubric_text = read_document(rubric.filename, await rubric.read())
        notes_text = read_document(notes.filename, await notes.read()) if notes else ""
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not rubric_text.strip():
        raise HTTPException(400, "Rubric file is empty.")
    r, n = store.ingest(rubric_id, rubric_text, notes_text, rubric_source=rubric.filename,
                        notes_source=notes.filename if notes else "notes")
    return IngestResponse(rubric_id=rubric_id, rubric_chunks=r, notes_chunks=n)


@app.post("/grade", response_model=GradeResponse)
def grade(req: GradeRequest, store: Store = Depends(get_store), llm: grader.LLM = Depends(get_llm)):
    """Grade an answer against the top-4 retrieved rubric chunks.

    Errors: 404 unknown rubric_collection, 422 invalid request body, 500 server misconfigured
    (e.g. missing API key), 502 LLM provider failed or kept returning invalid output.
    """
    try:
        return grader.grade(store, req.rubric_collection, req.question, req.answer, llm=llm)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except grader.LLMConfigError as e:
        raise HTTPException(500, f"Server configuration error: {e}")
    except grader.LLMProviderError as e:
        raise HTTPException(502, f"LLM provider error: {e}")
    except grader.LLMOutputError as e:
        raise HTTPException(502, f"Grading failed: {e}")
