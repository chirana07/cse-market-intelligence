from __future__ import annotations

import pandas as pd
import streamlit as st

from src.db.repositories import get_agent_run_counts, get_document_counts
from src.observability import load_recent_agent_runs
from src.settings import SETTINGS
from src.ui import divider_label, inject_global_styles, page_header


inject_global_styles()

page_header(
    "Agent Observability",
    "Inspect analyst runs, grounding decisions, evidence quality, and tool trajectory.",
)

try:
    document_counts = get_document_counts()
    run_counts = get_agent_run_counts()
    db_status = "Connected"
except Exception as exc:
    document_counts = {"documents": 0, "chunks": 0}
    run_counts = {"runs": 0, "blocked": 0, "caution": 0}
    db_status = f"Unavailable: {exc}"

storage = st.columns(4)
storage[0].metric("SQLite", db_status)
storage[1].metric("Indexed Docs", document_counts["documents"])
storage[2].metric("Indexed Chunks", document_counts["chunks"])
storage[3].metric("Persisted Runs", run_counts["runs"])
st.caption(f"Database: `{SETTINGS.sqlite_path}` · JSONL fallback: `{SETTINGS.log_path}`")

runs = load_recent_agent_runs(limit=200)

if not runs:
    st.info(
        "No agent runs logged yet. Run an analysis in AI Analyst Copilot to populate "
        f"`{SETTINGS.log_path}`."
    )
    st.stop()

summary_rows = []
for run in runs:
    metrics = run.get("evidence_metrics", {}) or {}
    critic = run.get("critic", {}) or {}
    output_validation = run.get("output_validation", {}) or {}
    request = run.get("request", {}) or {}
    summary_rows.append(
        {
            "logged_at": run.get("logged_at", ""),
            "run_id": run.get("run_id", ""),
            "mode": request.get("analysis_mode", ""),
            "ticker": request.get("ticker", ""),
            "latency_sec": run.get("latency_sec", ""),
            "critic": critic.get("status", ""),
            "confidence": critic.get("confidence", ""),
            "output": output_validation.get("status", ""),
            "format_score": output_validation.get("structure_score", ""),
            "evidence_score": metrics.get("evidence_score", 0),
            "sources": metrics.get("unique_source_count", 0),
            "chunks": metrics.get("retrieved_chunk_count", 0),
        }
    )

summary_df = pd.DataFrame(summary_rows)

top = st.columns(4)
top[0].metric("Logged Runs", len(summary_df))
top[1].metric("Avg Evidence", f"{summary_df['evidence_score'].mean():.1f}")
top[2].metric("Avg Latency", f"{pd.to_numeric(summary_df['latency_sec'], errors='coerce').mean():.2f}s")
top[3].metric("Blocked/Caution", int(summary_df["critic"].isin(["blocked", "caution"]).sum()))

divider_label("Run Table")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

selected_run_id = st.selectbox("Inspect run", summary_df["run_id"].tolist())
selected = next((run for run in runs if run.get("run_id") == selected_run_id), runs[0])

divider_label("Run Details")
detail_col1, detail_col2 = st.columns(2)
with detail_col1:
    st.markdown("#### Request")
    st.json(selected.get("request", {}))
    st.markdown("#### Evidence Metrics")
    st.json(selected.get("evidence_metrics", {}))
    st.markdown("#### Output Validation")
    st.json(selected.get("output_validation", {}))

with detail_col2:
    st.markdown("#### Grounding Critic")
    st.json(selected.get("critic", {}))
    st.markdown("#### Sources")
    st.json(selected.get("sources", []))

with st.expander("Agent Trajectory", expanded=True):
    st.json(selected.get("trajectory", []))
