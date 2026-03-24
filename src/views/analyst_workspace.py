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
from src.rag_chain import build_qa_chain
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

page_header(
    "AI Analyst Copilot",
    "Ask deep research questions, synthesize evidence, and generate analyst-grade memos.",
)


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
# ----------------------st.sidebar.header("📰 News Sources")

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
        enriched_question = f"""
Company Focus: {company_name or "Not specified"}
Ticker Focus: {ticker or selected_ticker}
Analysis Mode: {analysis_mode}
Detected Ticker Filter: {selected_ticker}
Event Filter: {selected_event}
Domain Filter: {selected_domain}
Source Filter: {selected_source}

Research Question:
{research_query}
""".strip()

        with st.spinner("Generating analyst output..."):
            chain = build_qa_chain(
                vectorstore,
                domain_filter=selected_domain,
                source_filter=selected_source,
            )
            result = chain.invoke({"question": enriched_question})

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
        "💾 Download Memo (.md)",
        data=st.session_state.analysis_memo_md,
        file_name=st.session_state.analysis_memo_filename,
        mime="text/markdown",
        use_container_width=True,
    )

    if export_col2.button("📌 Save to History", use_container_width=True):
        cache_key = build_memo_cache_key(
            meta.get("company_name", ""),
            meta.get("ticker", ""),
            meta.get("analysis_mode", ""),
            meta.get("research_query", "")
        )
        save_memo_artifact(cache_key, st.session_state.analysis_memo_md, meta)
        st.toast("Memo saved to history!")

    export_col3.caption(
        f"Mode: **{meta.get('analysis_mode', 'N/A')}** · "
        f"Domain: {meta.get('selected_domain', 'All')} · "
        f"Ticker: {meta.get('selected_ticker', 'All')}"
    )

    source_docs = result.get("source_documents", [])
    if source_docs:
        formatted_docs = format_evidence(source_docs)
        metrics = compute_retrieval_metrics(formatted_docs)

        divider_label("Evidence Quality")

        ev_c1, ev_c2, ev_c3, ev_c4 = st.columns(4)
        ev_c1.metric("Confidence", metrics["confidence_label"])
        ev_c2.metric("Coverage", metrics["coverage_label"])
        ev_c3.metric("Sources", metrics["unique_source_count"])
        ev_c4.metric("Score", f"{metrics['evidence_score']}/100")

        if metrics["gaps_or_warnings"]:
            for w in metrics["gaps_or_warnings"]:
                st.warning(w)

        if metrics["confidence_label"] == "Low":
            st.info("Limited evidence diversity — treat this answer with caution.")
        elif metrics["confidence_label"] == "High":
            st.success("Strong evidence grounding — answer is well-supported.")

        with st.expander("🧠 Grade Answer Support (AI Judge)"):
            if st.checkbox("Run AI grading pass", value=False):
                with st.spinner("Grading..."):
                    grade = grade_answer_support(meta.get("research_query", ""), answer_text, formatted_docs)
                st.markdown(f"**Support Grade:** `{grade}`")

        with st.expander("📎 Evidence Sources ({} chunks)".format(len(formatted_docs))):
            for i, doc in enumerate(formatted_docs, start=1):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**{i}. {doc['title']}**")
                    c2.caption(doc['domain'])
                    st.caption(f"🔗 {doc['source_url']}")
                    if doc["tickers"]:
                        st.caption(f"Tickers: {doc['tickers']}")
                    if doc["events"]:
                        st.caption(f"Events: {doc['events']}")
                    st.write(doc["snippet"])
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
        with st.expander(f"📝 {m_title} · {date_str}"):
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