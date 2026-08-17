from __future__ import annotations
import pandas as pd
import streamlit as st

from src.config import VECTORSTORE_DIR
from src.loaders import load_urls, parse_uploaded_txt_file
from src.splitter import split_documents
from src.vectorstore import (
    clear_vectorstore,
    get_vectorstore_stats,
    ingest_chunks,
    load_vectorstore,
)
from src.agents.graph import run_analyst_workflow
from src.agents.state import AnalystRequest
from src.research_memo import build_memo_filename, build_research_memo_markdown
from src.persistence import (
    build_memo_cache_key,
    save_memo_artifact,
    load_recent_memos,
)
from src.rag_evaluation import compute_retrieval_metrics, grade_answer_support
from src.ui import inject_global_styles, page_header, section_header, info_card, status_badge, empty_state, divider_label, chat_message
from src.app_state import set_active_symbol
from src.evidence_formatter import format_evidence
from src.alerts_engine import DEFAULT_ALERTS_FILE, evaluate_alerts
from src.db.repositories import get_agent_run_counts, get_document_counts
from src.observability import load_recent_agent_runs
from src.settings import SETTINGS

inject_global_styles()

# ---------------------------
# Session state init
# ---------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_memo_md" not in st.session_state:
    st.session_state.analysis_memo_md = ""

if "analysis_memo_filename" not in st.session_state:
    st.session_state.analysis_memo_filename = "equity_research_memo.md"

if "analysis_meta" not in st.session_state:
    st.session_state.analysis_meta = {}

if "company_name_input" not in st.session_state:
    st.session_state.company_name_input = ""

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = ""

if "analysis_mode_input" not in st.session_state:
    st.session_state.analysis_mode_input = "News Summary"

if "research_query_input" not in st.session_state:
    st.session_state.research_query_input = ""

# Prefill from Market Dashboard
pending = st.session_state.pop("pending_market_selection", None)
if pending:
    st.session_state.company_name_input = pending.get("company_name", "")
    st.session_state.ticker_input = pending.get("ticker", "")
    st.session_state.analysis_mode_input = pending.get("analysis_mode", "News Summary")
    st.session_state.research_query_input = pending.get("query", "")

if st.session_state.get("ticker_input") or st.session_state.get("company_name_input"):
    set_active_symbol(
        st.session_state.get("ticker_input", ""),
        st.session_state.get("company_name_input", ""),
    )

def render_copilot():

    # ---------------------------
    # Helpers
    # ---------------------------
    def doc_matches_filters(doc, domain_filter="All", ticker_filter="All", event_filter="All"):
        if domain_filter != "All" and doc["domain"] != domain_filter:
            return False
        if ticker_filter != "All" and ticker_filter not in doc.get("tickers", []):
            return False
        if event_filter != "All" and event_filter not in doc.get("event_tags", []):
            return False
        return True


    # ---------------------------
    # Sidebar: ingestion
    # ---------------------------
    st.sidebar.header("Source Ingestion")

    manual_url_1 = st.sidebar.text_input("URL 1")
    manual_url_2 = st.sidebar.text_input("URL 2")
    manual_url_3 = st.sidebar.text_input("URL 3")

    st.sidebar.subheader("Offline Batch Processing")
    uploaded_txt = st.sidebar.file_uploader(
        "Upload a .txt file containing URLs or mixed notes with embedded URLs",
        type=["txt"],
    )

    manual_urls = [manual_url_1, manual_url_2, manual_url_3]
    has_manual_urls = any(url.strip() for url in manual_urls)
    has_uploaded_file = uploaded_txt is not None

    process_manual = st.sidebar.button(
        "Process Manual URLs",
        disabled=not has_manual_urls,
    )

    process_batch = st.sidebar.button(
        "Batch Process",
        disabled=not has_uploaded_file,
    )

    reset_index = st.sidebar.button("Reset Index")

    if reset_index:
        clear_vectorstore(VECTORSTORE_DIR)
        st.sidebar.success("Index cleared.")
        st.session_state.analysis_result = None
        st.session_state.analysis_memo_md = ""
        st.session_state.analysis_meta = {}
        st.rerun()


    # ---------------------------
    # Current stats
    # ---------------------------
    stats = get_vectorstore_stats(VECTORSTORE_DIR)

    st.sidebar.divider()
    st.sidebar.subheader("Index Status")
    st.sidebar.caption(f"{stats['chunk_count']} chunks · {stats['source_count']} sources · {stats['domain_count']} domains")
    st.sidebar.caption(f"{len(stats['tickers'])} tickers · {len(stats['event_tags'])} event tags")

    if stats["documents"]:
        with st.sidebar.expander("Source Catalog"):
            for doc in stats["documents"]:
                st.markdown(f"**{doc['title']}**")
                st.caption(f"{doc['domain']} · {doc['source']}")
                if doc["tickers_str"]:
                    st.caption(f"Tickers: {doc['tickers_str']}")
                if doc["event_tags_str"]:
                    st.caption(f"Events: {doc['event_tags_str']}")


    # ---------------------------
    # Process manual URLs
    # ---------------------------
    if process_manual:
        with st.spinner("Loading manual URLs..."):
            docs, failed_urls = load_urls(manual_urls)

        if failed_urls:
            st.warning("These manual URLs could not be loaded:")
            for url in failed_urls:
                st.write(f"- {url}")

        if not docs:
            st.error("No valid content loaded from the manual URLs.")
        else:
            with st.spinner("Splitting documents..."):
                chunks = split_documents(docs)

            with st.spinner("Updating vector database..."):
                _, ingest_stats = ingest_chunks(chunks, VECTORSTORE_DIR)

            st.success(
                f"Manual ingest complete. "
                f"New chunks: {ingest_stats['new_chunks']}, "
                f"duplicates skipped: {ingest_stats['skipped_duplicates']}."
            )
            st.rerun()


    # ---------------------------
    # Process batch file
    # ---------------------------
    if process_batch:
        batch_urls, invalid_lines = parse_uploaded_txt_file(uploaded_txt)

        if uploaded_txt is None:
            st.error("Please upload a .txt file first.")
        else:
            if invalid_lines:
                st.info(
                    f"Ignored {len(invalid_lines)} non-URL line(s). "
                    f"URLs were still extracted from the file where possible."
                )
                with st.expander("Show ignored lines"):
                    for line in invalid_lines[:20]:
                        st.write(line)

            if not batch_urls:
                st.error("No valid URLs were found in the uploaded text file.")
            else:
                st.info(f"Found {len(batch_urls)} valid URL(s) in batch file.")

                with st.spinner("Loading batch URLs..."):
                    docs, failed_urls = load_urls(batch_urls)

                if failed_urls:
                    st.warning("These batch URLs could not be loaded:")
                    for url in failed_urls:
                        st.write(f"- {url}")

                if not docs:
                    st.error("No valid content loaded from the batch URLs.")
                else:
                    with st.spinner("Splitting documents..."):
                        chunks = split_documents(docs)

                    with st.spinner("Updating vector database..."):
                        _, ingest_stats = ingest_chunks(chunks, VECTORSTORE_DIR)

                    st.success(
                        f"Batch ingest complete. "
                        f"New chunks: {ingest_stats['new_chunks']}, "
                        f"duplicates skipped: {ingest_stats['skipped_duplicates']}."
                    )
                    st.rerun()


    # refresh stats after rerun-sensitive actions
    stats = get_vectorstore_stats(VECTORSTORE_DIR)


    # ---------------------------
    # Analyst workspace
    # ---------------------------
    divider_label("Research Configuration")

    with st.container(border=True):
        top_col1, top_col2, top_col3 = st.columns(3)

        company_name = top_col1.text_input(
            "Company",
            key="company_name_input",
            placeholder="e.g. John Keells Holdings",
        )

        ticker = top_col2.text_input(
            "Ticker",
            key="ticker_input",
            placeholder="e.g. JKH.N0000",
        )

        analysis_mode = top_col3.selectbox(
            "Analysis Mode",
            [
                "News Summary",
                "Bull vs Bear Case",
                "Catalysts & Risks",
                "Earnings Impact",
                "Strategy / Management Signals",
                "Portfolio Memo",
            ],
            key="analysis_mode_input",
        )

        with st.expander("Advanced Filters"):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            selected_domain = filter_col1.selectbox("Filter by domain", ["All"] + stats["domains"])
            selected_ticker = filter_col2.selectbox("Filter by detected ticker", ["All"] + stats["tickers"])
            selected_event = filter_col3.selectbox("Filter by event tag", ["All"] + stats["event_tags"])

            filtered_docs = [
                doc for doc in stats["documents"]
                if doc_matches_filters(doc, selected_domain, selected_ticker, selected_event)
            ]

            selected_source = st.selectbox(
                "Filter by source URL",
                ["All"] + [doc["source"] for doc in filtered_docs]
            )
        if "selected_domain" not in dir():
            selected_domain, selected_ticker, selected_event, selected_source = "All", "All", "All", "All"
            filtered_docs = list(stats["documents"])

        research_query = st.text_area(
            "Research question",
            key="research_query_input",
            placeholder="e.g. What are the earnings outlook implications and key risks from these disclosures?",
            height=100,
            label_visibility="collapsed",
        )

        run_col1, run_col2, run_col3 = st.columns([2, 1, 1])
        run_analysis = run_col1.button("Run Analysis", use_container_width=True, type="primary")
        run_col2.metric("Sources", len(filtered_docs))
        run_col3.metric("Domains", len(sorted({doc["domain"] for doc in filtered_docs})))

    if run_analysis:
        vectorstore = load_vectorstore(VECTORSTORE_DIR)

        if vectorstore is None:
            st.warning("Please process URLs first.")
        elif not research_query.strip():
            st.warning("Please enter a research question.")
        else:
            with st.spinner("Generating analyst output..."):
                request = AnalystRequest(
                    company_name=company_name,
                    ticker=ticker or selected_ticker,
                    analysis_mode=analysis_mode,
                    research_query=research_query,
                    selected_domain=selected_domain,
                    selected_source=selected_source,
                    selected_ticker=selected_ticker,
                    selected_event=selected_event,
                )
                result = run_analyst_workflow(vectorstore, request)

            memo_md = build_research_memo_markdown(
                company_name=company_name,
                ticker=ticker or selected_ticker,
                analysis_mode=analysis_mode,
                user_query=research_query,
                answer=result.get("answer", ""),
                source_docs=result.get("source_documents", []),
                selected_domain=selected_domain,
                selected_source=selected_source,
                selected_ticker=selected_ticker,
                selected_event=selected_event,
            )

            st.session_state.analysis_result = result
            st.session_state.analysis_memo_md = memo_md
            st.session_state.analysis_memo_filename = build_memo_filename(
                company_name=company_name,
                ticker=ticker or selected_ticker,
                analysis_mode=analysis_mode,
            )
            st.session_state.analysis_meta = {
                "run_id": result.get("run_id", ""),
                "company_name": company_name,
                "ticker": ticker or selected_ticker,
                "analysis_mode": analysis_mode,
                "selected_domain": selected_domain,
                "selected_source": selected_source,
                "selected_ticker": selected_ticker,
                "selected_event": selected_event,
                "research_query": research_query,
            }


    # ---------------------------
    # Render latest analysis
    # ---------------------------
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        meta = st.session_state.analysis_meta

        divider_label("Analyst Output")

        # User query bubble
        if meta.get("research_query"):
            chat_message("user", meta["research_query"])

        # AI answer bubble
        answer_text = result.get("answer", "No answer generated.")
        chat_message("assistant", answer_text)

        # Export actions
        st.markdown("<br>", unsafe_allow_html=True)
        export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
        export_col1.download_button(
            "Download Memo",
            data=st.session_state.analysis_memo_md,
            file_name=st.session_state.analysis_memo_filename,
            mime="text/markdown",
            use_container_width=True,
        )

        if export_col2.button("Save to History", use_container_width=True):
            cache_key = build_memo_cache_key(
                meta.get("company_name", ""),
                meta.get("ticker", ""),
                meta.get("analysis_mode", ""),
                meta.get("research_query", "")
            )
            save_memo_artifact(cache_key, st.session_state.analysis_memo_md, meta)
            st.toast("Memo saved to history!")

        export_col3.caption(
            f"Run: `{meta.get('run_id', 'N/A')}` · "
            f"Mode: **{meta.get('analysis_mode', 'N/A')}** · "
            f"Domain: {meta.get('selected_domain', 'All')} · "
            f"Ticker: {meta.get('selected_ticker', 'All')}"
        )

        source_docs = result.get("source_documents", [])
        if source_docs:
            formatted_docs = format_evidence(source_docs)
            metrics = result.get("evidence_metrics") or compute_retrieval_metrics(formatted_docs)
            critic = result.get("critic") or {}

            divider_label("Evidence Quality")

            ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
            ev_c1.metric("Confidence", metrics["confidence_label"])
            ev_c2.metric("Coverage", metrics["coverage_label"])
            ev_c3.metric("Sources", metrics["unique_source_count"])
            ev_c4.metric("Score", f"{metrics['evidence_score']}/100")

            if metrics["gaps_or_warnings"]:
                for w in metrics["gaps_or_warnings"]:
                    st.warning(w)

            if critic:
                st.caption(
                    f"Grounding critic: **{critic.get('status', 'unknown')}** · "
                    f"{critic.get('message', '')}"
                )

            if metrics["confidence_label"] == "Low":
                st.info("Limited evidence diversity — treat this answer with caution.")
            elif metrics["confidence_label"] == "High":
                st.success("Strong evidence grounding — answer is well-supported.")

            with st.expander("Grade Answer Support"):
                if st.checkbox("Run AI grading pass", value=False):
                    with st.spinner("Grading..."):
                        grade = grade_answer_support(meta.get("research_query", ""), answer_text, formatted_docs)
                    st.markdown(f"**Support Grade:** `{grade}`")

            with st.expander("Evidence Sources ({} chunks)".format(len(formatted_docs))):
                for i, doc in enumerate(formatted_docs, start=1):
                    with st.container(border=True):
                        c1, c2 = st.columns([3, 1])
                        c1.markdown(f"**{i}. {doc['title']}**")
                        c2.caption(doc['domain'])
                        st.caption(f"Source: {doc['source_url']}")
                        if doc["tickers"]:
                            st.caption(f"Tickers: {doc['tickers']}")
                        if doc["events"]:
                            st.caption(f"Events: {doc['events']}")
                        st.write(doc["snippet"])
        else:
            critic = result.get("critic") or {}
            if critic:
                divider_label("Evidence Quality")
                st.warning(critic.get("message", "No retrieved evidence was available."))

        trajectory = result.get("trajectory", [])
        if trajectory:
            with st.expander("Agent Trace"):
                st.json(trajectory)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Enter a research question above and click **Run Analysis** to generate AI analyst output.")


    # ---------------------------
    # Recent Analyst Memos
    # ---------------------------
    divider_label("Recent Analyst Memos")

    recent_memos = load_recent_memos(limit=5)
    if recent_memos:
        for m in recent_memos:
            m_meta = m["meta"]
            m_title = m_meta.get("company_name") or m_meta.get("ticker") or "General"
            date_str = m_meta.get("created_at", "")[:10]
            with st.expander(f"{m_title} · {date_str}"):
                st.caption(f"**Query**: {m_meta.get('research_query')}")
                mode_badge = status_badge(m_meta.get('analysis_mode', 'N/A'), 'info')
                st.markdown(mode_badge, unsafe_allow_html=True)
                st.download_button(
                    "Download Memo",
                    data=m["memo"],
                    file_name=f"{m_title.replace(' ', '_')}_{date_str}.md",
                    mime="text/markdown",
                    key=f"dl_memo_{m['key']}"
                )
    else:
        st.info("No memos in history yet. Generate and save a memo above.")


    # ---------------------------
    # Catalog + Benchmark (collapsed)
    # ---------------------------
    with st.expander("Indexed Source Catalog"):
        st.caption("Ticker and event extraction are heuristic aids, not authoritative labels.")
        if stats["documents"]:
            catalog_df = pd.DataFrame(filtered_docs if filtered_docs else stats["documents"])
            avail_cols = [c for c in ["title", "domain", "primary_ticker", "primary_event", "tickers_str", "event_tags_str", "source", "ingested_at"] if c in catalog_df.columns]
            st.dataframe(catalog_df[avail_cols], use_container_width=True)
        else:
            st.info("No indexed sources yet.")

    with st.expander("Benchmark / Eval Summary"):
        from src.persistence import load_latest_benchmark
        latest_eval = load_latest_benchmark()
        if latest_eval:
            st.caption(f"Latest run: {latest_eval.get('timestamp')}")
            eval_c1, eval_c2, eval_c3, eval_c4 = st.columns(4)
            eval_c1.metric("Total Cases", latest_eval.get("total_cases", 0))
            eval_c2.metric("Pass Rate", f"{latest_eval.get('pass_rate_pct', 0)}%")
            labels = latest_eval.get("label_counts", {})
            eval_c3.metric("Strong / Acceptable", f"{labels.get('Strong', 0)} / {labels.get('Acceptable', 0)}")
            eval_c4.metric("Weak", labels.get("Weak", 0))
            with st.expander("Case results"):
                st.json(latest_eval.get("case_results", []))
        else:
            st.caption("No benchmark history. Run `python scripts/run_benchmarks.py` to evaluate.")


page_header(
    "Copilot & Monitoring Hub",
    "Synthesize research with AI, manage alert rules, and inspect system observability.",
)

cp_tab1, cp_tab2, cp_tab3 = st.tabs(["AI Analyst Copilot", "Alert Rules & Monitoring", "System Observability"])

with cp_tab1:
    render_copilot()

with cp_tab2:
    st.subheader("Alert Rules & Triggered Alerts")
    try:
        alerts_df_raw, triggered_df_raw = evaluate_alerts(
            universe_path=Path("data/cse_universe.csv"),
            file_path=DEFAULT_ALERTS_FILE,
        )
    except Exception:
        alerts_df_raw, triggered_df_raw = pd.DataFrame(), pd.DataFrame()

    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Configured Alert Rules", len(alerts_df_raw) if isinstance(alerts_df_raw, pd.DataFrame) else 0)
    m_col2.metric("Triggered Now", len(triggered_df_raw) if isinstance(triggered_df_raw, pd.DataFrame) else 0)

    if isinstance(triggered_df_raw, pd.DataFrame) and not triggered_df_raw.empty:
        st.markdown("#### Currently Triggered Alerts")
        st.dataframe(triggered_df_raw, use_container_width=True, hide_index=True)

    st.markdown("#### Active Alert Rules")
    if isinstance(alerts_df_raw, pd.DataFrame) and not alerts_df_raw.empty:
        st.dataframe(alerts_df_raw, use_container_width=True, hide_index=True)
    else:
        empty_state("No Alert Rules", "No alerts configured yet.", "Use the rule builder to set up price and volume alerts.")

with cp_tab3:
    st.subheader("System & Agent Observability")
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

    recent_runs = load_recent_agent_runs(limit=50)
    if recent_runs:
        st.markdown("#### Recent Agent Runs")
        obs_df = pd.DataFrame([
            {
                "logged_at": r.get("logged_at", ""),
                "run_id": r.get("run_id", ""),
                "mode": (r.get("request") or {}).get("analysis_mode", ""),
                "ticker": (r.get("request") or {}).get("ticker", ""),
                "latency_sec": r.get("latency_sec", ""),
                "critic": (r.get("critic") or {}).get("status", ""),
            }
            for r in recent_runs
        ])
        st.dataframe(obs_df, use_container_width=True, hide_index=True)
    else:
        empty_state("No Agent Runs", "No persisted runs found in SQLite or log file.")
