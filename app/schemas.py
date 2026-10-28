"""Pydantic models shared by the API, the grader and the UI."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    similarity: float | None = None  # cosine similarity to the query, set by retrieval


class IngestRequest(BaseModel):
    rubric_id: str = Field(..., min_length=1, description="Name for this rubric, e.g. 'photosynthesis-q1'")
    rubric_text: str = Field(..., min_length=1)
    notes_text: str = ""


class IngestResponse(BaseModel):
    rubric_id: str
    rubric_chunks: int
    notes_chunks: int


class GradeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    rubric_collection: str = Field(..., min_length=1, description="Name the rubric was ingested under")


class CriterionResult(BaseModel):
    name: str = Field(..., min_length=1)
    met: bool
    feedback: str
    cited_chunk_ids: list[str] = Field(default_factory=list, description="Retrieved chunk ids this is based on")


class GradeResponse(BaseModel):
    """What the LLM must return, and what /grade returns after our checks."""

    score: int = Field(..., ge=0)
    max_score: int = Field(..., gt=0)
    criteria: list[CriterionResult] = Field(..., min_length=1)
    overall_feedback: str
    needs_human_review: bool = False
    reasons: list[str] = Field(default_factory=list, description="Why human review is needed")

    @model_validator(mode="after")
    def score_not_above_max(self):
        if self.score > self.max_score:
            raise ValueError(f"score ({self.score}) cannot be greater than max_score ({self.max_score})")
        return self
