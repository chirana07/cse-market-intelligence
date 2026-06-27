from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from src.agents.graph import run_analyst_workflow
from src.agents.state import AnalystRequest
from src.api.schemas import HealthResponse, ResearchQueryRequest, ResearchQueryResponse
from src.db.repositories import get_agent_run_counts, get_document_counts
from src.observability import load_recent_agent_runs
from src.settings import SETTINGS
from src.vectorstore import load_vectorstore


app = FastAPI(
    title="CSE Market Intelligence API",
    version="1.0.0",
    description="Production API surface for CSE RAG and agentic equity research workflows.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    vectorstore = load_vectorstore(SETTINGS.vectorstore_dir)
    document_counts = get_document_counts()
    run_counts = get_agent_run_counts()
    return HealthResponse(
        status="ok",
        vectorstore_exists=vectorstore is not None,
        documents=document_counts["documents"],
        chunks=document_counts["chunks"],
        persisted_runs=run_counts["runs"],
    )


@app.post("/research/query", response_model=ResearchQueryResponse)
def research_query(payload: ResearchQueryRequest) -> ResearchQueryResponse:
    vectorstore = load_vectorstore(SETTINGS.vectorstore_dir)
    if vectorstore is None:
        raise HTTPException(
            status_code=409,
            detail="No vectorstore is available. Ingest sources in the Streamlit app before querying the API.",
        )

    request = AnalystRequest(
        company_name=payload.company_name,
        ticker=payload.ticker,
        analysis_mode=payload.analysis_mode,
        research_query=payload.question,
        selected_domain=payload.selected_domain,
        selected_source=payload.selected_source,
        selected_ticker=payload.selected_ticker,
        selected_event=payload.selected_event,
    )
    result = run_analyst_workflow(vectorstore, request)
    return ResearchQueryResponse(
        run_id=result["run_id"],
        answer=result["answer"],
        evidence_metrics=result.get("evidence_metrics", {}),
        critic=result.get("critic", {}),
        output_validation=result.get("output_validation", {}),
        source_count=len(result.get("source_documents", [])),
        trajectory=result.get("trajectory", []),
    )


@app.get("/observability/runs")
def recent_runs(limit: int = Query(default=25, ge=1, le=200)):
    return {"runs": load_recent_agent_runs(limit=limit)}
