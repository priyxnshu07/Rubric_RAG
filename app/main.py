"""FastAPI backend. Run with: uvicorn app.main:app --reload"""
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from app import grader
from app.loaders import read_document
from app.schemas import GradeRequest, GradeResponse, IngestRequest, IngestResponse
from app.store import Store

app = FastAPI(title="RubricRAG", description="Rubric-grounded feedback assistant")


@lru_cache
def get_store() -> Store:
    return Store()  # created once; loading the embedding model is slow


def get_llm() -> grader.LLM:
    return grader.default_llm()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/rubrics", response_model=list[str])
def list_rubrics(store: Store = Depends(get_store)):
    return store.list_rubrics()


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
    try:
        return grader.grade(store, req.rubric_id, req.answer, req.question, llm=llm)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:  # e.g. missing API key
        raise HTTPException(500, str(e))
    except ValueError as e:  # LLM output could not be parsed
        raise HTTPException(502, str(e))
