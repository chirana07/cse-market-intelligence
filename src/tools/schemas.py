from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ToolError(BaseModel):
    tool_name: str
    error_type: str
    message: str
    retryable: bool = False
    attempts: int = 1
    elapsed_sec: float | None = None


class RetrievalInput(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    selected_domain: str = "All"
    selected_source: str = "All"
    selected_ticker: str = "All"
    selected_event: str = "All"


class RetrievalEvidenceOutput(BaseModel):
    formatted_docs: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class MarketQuoteInput(BaseModel):
    symbol: str = Field(min_length=1, max_length=32)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class MarketQuoteOutput(BaseModel):
    requested_symbol: str
    canonical_symbol: str
    company_name: str | None = None
    currency: str | None = None
    last_traded_price: float | None = None
    change_pct: float | None = None
    source: Literal["yahoo"] = "yahoo"
    attempts: int = 1
    elapsed_sec: float | None = None


class DisclosureFetchInput(BaseModel):
    category: str = "All"
    limit: int = Field(default=25, ge=1, le=200)


class DisclosureFetchOutput(BaseModel):
    rows: list[dict[str, Any]]
    row_count: int
    source: Literal["cse"] = "cse"
    attempts: int = 1
    elapsed_sec: float | None = None
