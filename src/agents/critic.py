from __future__ import annotations

from typing import Any

from src.settings import SETTINGS


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I do not have sufficient retrieved context to answer this question. "
    "Please ingest more relevant CSE announcements, reports, or news sources and try again."
)


def evaluate_grounding(answer: str, evidence_metrics: dict[str, Any]) -> dict[str, Any]:
    evidence_score = int(evidence_metrics.get("evidence_score", 0) or 0)
    chunk_count = int(evidence_metrics.get("retrieved_chunk_count", 0) or 0)
    warnings = list(evidence_metrics.get("gaps_or_warnings", []) or [])
    answer_text = (answer or "").strip()
    lower_answer = answer_text.lower()

    says_insufficient = (
        "not enough information" in lower_answer
        or "insufficient retrieved context" in lower_answer
        or "do not have sufficient" in lower_answer
    )

    if chunk_count == 0:
        return {
            "status": "blocked",
            "confidence": "Low",
            "should_replace_answer": not says_insufficient,
            "message": "No retrieved evidence was available.",
            "warnings": warnings,
        }

    if evidence_score < SETTINGS.min_evidence_score:
        return {
            "status": "caution",
            "confidence": "Low",
            "should_replace_answer": False,
            "message": "Retrieved evidence is limited; answer should be treated as a low-confidence synthesis.",
            "warnings": warnings,
        }

    return {
        "status": "approved",
        "confidence": evidence_metrics.get("confidence_label", "Medium"),
        "should_replace_answer": False,
        "message": "Answer passed the evidence availability gate.",
        "warnings": warnings,
    }


def apply_grounding_gate(answer: str, critic: dict[str, Any]) -> str:
    if critic.get("should_replace_answer"):
        return INSUFFICIENT_EVIDENCE_MESSAGE

    if critic.get("status") == "caution" and answer.strip():
        caveat = (
            "Evidence caveat: retrieved support is limited, so treat this as a preliminary "
            "view rather than a fully grounded conclusion.\n\n"
        )
        if answer.startswith("Evidence caveat:"):
            return answer
        return caveat + answer

    return answer

