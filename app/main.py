"""FastAPI backend. Run with: uvicorn app.main:app --reload"""
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile

from app import grader
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
    """Same as /ingest, but takes .txt / .md file uploads."""
    try:
        rubric_text = (await rubric.read()).decode("utf-8")
        notes_text = (await notes.read()).decode("utf-8") if notes else ""
    except UnicodeDecodeError:
        raise HTTPException(400, "Files must be UTF-8 text (.txt or .md).")
    if not rubric_text.strip():
        raise HTTPException(400, "Rubric file is empty.")
    return ingest(IngestRequest(rubric_id=rubric_id, rubric_text=rubric_text, notes_text=notes_text), store)


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
