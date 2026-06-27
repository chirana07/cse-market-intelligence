from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AnalystRequest:
    company_name: str = ""
    ticker: str = ""
    analysis_mode: str = "News Summary"
    research_query: str = ""
    selected_domain: str = "All"
    selected_source: str = "All"
    selected_ticker: str = "All"
    selected_event: str = "All"

    def enriched_question(self) -> str:
        return f"""
Company Focus: {self.company_name or "Not specified"}
Ticker Focus: {self.ticker or self.selected_ticker}
Analysis Mode: {self.analysis_mode}
Detected Ticker Filter: {self.selected_ticker}
Event Filter: {self.selected_event}
Domain Filter: {self.selected_domain}
Source Filter: {self.selected_source}

Research Question:
{self.research_query}
""".strip()


@dataclass
class AnalystRunResult:
    run_id: str
    answer: str
    source_documents: list[Any] = field(default_factory=list)
    evidence_metrics: dict[str, Any] = field(default_factory=dict)
    critic: dict[str, Any] = field(default_factory=dict)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def to_chain_result(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "answer": self.answer,
            "source_documents": self.source_documents,
            "evidence_metrics": self.evidence_metrics,
            "critic": self.critic,
            "trajectory": self.trajectory,
        }

