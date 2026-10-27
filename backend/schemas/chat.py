"""Request and response schemas for the RAG API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Bound input size prevents accidental or abusive oversized prompts.
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    # These fields mirror the dictionaries returned by Ask.retrieve().
    text: str
    source: str
    page: int
    distance: float


class ChatResponse(BaseModel):
    # Return both the answer and evidence so the UI can show paper sources.
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
