"""Pydantic models shared by the API, the grader and the UI."""
from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One retrievable piece of a rubric or course notes."""

    id: str  # e.g. "R7" (rubric line 7), "R9-13" (rubric lines 9-13) or "N3-5" (notes lines 3-5)
    text: str
    doc_type: Literal["rubric", "notes"]
    start_line: int
    end_line: int
    source: str = ""  # file the chunk came from
    criterion: str = ""  # e.g. "Q1 Photosynthesis > Inputs"; empty for notes and general rubric text
    points: float | None = None  # max points of the criterion, if stated


class IngestRequest(BaseModel):
    rubric_id: str = Field(..., min_length=1, description="Name for this rubric, e.g. 'photosynthesis-q1'")
    rubric_text: str = Field(..., min_length=1)
    notes_text: str = ""


class IngestResponse(BaseModel):
    rubric_id: str
    rubric_chunks: int
    notes_chunks: int


class GradeRequest(BaseModel):
    rubric_id: str
    question: str = ""
    answer: str = Field(..., min_length=1)


class CriterionResult(BaseModel):
    criterion: str
    score: float = Field(..., ge=0)
    max_score: float = Field(..., gt=0)
    feedback: str
    citations: list[str] = Field(default_factory=list, description="Chunk ids the judgement is based on")


class Citation(BaseModel):
    id: str
    text: str
    start_line: int
    end_line: int
    source: str = ""
    criterion: str = ""


class GradeResponse(BaseModel):
    rubric_id: str
    total_score: float
    max_score: float
    criteria: list[CriterionResult]
    citations: list[Citation]  # the cited chunks, resolved to their text and line numbers
    warnings: list[str] = Field(default_factory=list)
