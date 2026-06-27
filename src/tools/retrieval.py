from __future__ import annotations

from typing import Any

from src.evidence_formatter import format_evidence
from src.rag_evaluation import compute_retrieval_metrics
from src.tools.schemas import RetrievalEvidenceOutput, RetrievalInput


def evaluate_retrieved_evidence(
    docs: list[Any],
    tool_input: RetrievalInput,
) -> RetrievalEvidenceOutput:
    formatted_docs = format_evidence(docs)
    metrics = compute_retrieval_metrics(formatted_docs)
    metrics["requested_filters"] = {
        "selected_domain": tool_input.selected_domain,
        "selected_source": tool_input.selected_source,
        "selected_ticker": tool_input.selected_ticker,
        "selected_event": tool_input.selected_event,
    }
    return RetrievalEvidenceOutput(formatted_docs=formatted_docs, metrics=metrics)

