from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResearchQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    company_name: str = ""
    ticker: str = ""
    analysis_mode: str = "News Summary"
    selected_domain: str = "All"
    selected_source: str = "All"
    selected_ticker: str = "All"
    selected_event: str = "All"


class ResearchQueryResponse(BaseModel):
    run_id: str
    answer: str
    evidence_metrics: dict[str, Any]
    critic: dict[str, Any]
    output_validation: dict[str, Any]
    source_count: int
    trajectory: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    vectorstore_exists: bool
    documents: int
    chunks: int
    persisted_runs: int
