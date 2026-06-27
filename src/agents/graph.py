from __future__ import annotations

import time
from typing import Any

from src.agents.critic import apply_grounding_gate, evaluate_grounding
from src.agents.state import AnalystRequest, AnalystRunResult
from src.observability import log_agent_run, new_run_id
from src.rag_chain import build_qa_chain
from src.tools.retrieval import evaluate_retrieved_evidence
from src.tools.schemas import RetrievalInput


def _source_log_summary(formatted_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for doc in formatted_docs:
        summary.append(
            {
                "title": doc.get("title"),
                "source_url": doc.get("source_url"),
                "domain": doc.get("domain"),
                "chunk_id": doc.get("chunk_id"),
                "tickers": doc.get("tickers"),
                "events": doc.get("events"),
            }
        )
    return summary


def run_analyst_workflow(
    vectorstore,
    request: AnalystRequest,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    run_id = run_id or new_run_id("analyst")
    trajectory: list[dict[str, Any]] = []
    started = time.time()

    trajectory.append(
        {
            "node": "intake_router",
            "status": "completed",
            "analysis_mode": request.analysis_mode,
            "has_company": bool(request.company_name),
            "has_ticker": bool(request.ticker or request.selected_ticker != "All"),
        }
    )

    enriched_question = request.enriched_question()
    trajectory.append(
        {
            "node": "research_planner",
            "status": "completed",
            "plan": [
                "Apply selected domain/source filters",
                "Retrieve CSE evidence chunks",
                "Generate structured analyst answer",
                "Score evidence quality",
                "Apply grounding gate",
            ],
        }
    )

    try:
        chain = build_qa_chain(
            vectorstore,
            domain_filter=request.selected_domain,
            source_filter=request.selected_source,
        )
        raw_result = chain.invoke({"question": enriched_question})
        answer = raw_result.get("answer", "")
        source_documents = raw_result.get("source_documents", [])
        trajectory.append(
            {
                "node": "rag_synthesis",
                "status": "completed",
                "source_document_count": len(source_documents),
            }
        )
    except Exception as exc:
        answer = f"Error during analysis: {exc}"
        source_documents = []
        trajectory.append(
            {
                "node": "rag_synthesis",
                "status": "failed",
                "error": str(exc),
            }
        )

    retrieval_input = RetrievalInput(
        question=request.research_query,
        selected_domain=request.selected_domain,
        selected_source=request.selected_source,
        selected_ticker=request.selected_ticker,
        selected_event=request.selected_event,
    )
    retrieval_output = evaluate_retrieved_evidence(source_documents, retrieval_input)
    formatted_docs = retrieval_output.formatted_docs
    evidence_metrics = retrieval_output.metrics
    trajectory.append(
        {
            "node": "evidence_retrieval",
            "status": "completed",
            "metrics": evidence_metrics,
        }
    )

    critic = evaluate_grounding(answer, evidence_metrics)
    answer = apply_grounding_gate(answer, critic)
    trajectory.append(
        {
            "node": "grounding_critic",
            "status": critic.get("status", "unknown"),
            "confidence": critic.get("confidence"),
            "message": critic.get("message"),
        }
    )

    elapsed = round(time.time() - started, 3)
    log_agent_run(
        {
            "run_id": run_id,
            "event_type": "analyst_workflow",
            "latency_sec": elapsed,
            "request": {
                "company_name": request.company_name,
                "ticker": request.ticker,
                "analysis_mode": request.analysis_mode,
                "selected_domain": request.selected_domain,
                "selected_source": request.selected_source,
                "selected_ticker": request.selected_ticker,
                "selected_event": request.selected_event,
                "research_query_len": len(request.research_query or ""),
            },
            "evidence_metrics": evidence_metrics,
            "critic": critic,
            "sources": _source_log_summary(formatted_docs),
            "trajectory": trajectory,
        }
    )

    return AnalystRunResult(
        run_id=run_id,
        answer=answer,
        source_documents=source_documents,
        evidence_metrics=evidence_metrics,
        critic=critic,
        trajectory=trajectory,
    ).to_chain_result()
