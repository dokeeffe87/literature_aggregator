"""Pydantic data models for papers, summaries, and rankings."""

from datetime import datetime

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Source-agnostic representation of an academic paper."""

    source: str
    source_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    url: str
    pdf_url: str | None = None
    categories: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """Claude-generated summary of a paper."""

    source_id: str
    summary: str = Field(description="2-3 sentence summary")
    key_contribution: str = Field(description="One-line key contribution")


class RankedPaper(BaseModel):
    """A paper with its summary and relevance ranking."""

    paper: Paper
    summary: PaperSummary
    relevance_score: float = Field(ge=0, le=10, description="Relevance score 0-10")
    relevance_reason: str
