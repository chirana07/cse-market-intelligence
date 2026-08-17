from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.report_intelligence import (
    compare_reports,
    extract_interim_key_figures,
    extract_pdf_text_from_bytes,
    extract_pdf_text_from_url,
    summarize_report,
)
from src.event_extraction import extract_events_from_report, event_importance_score
from src.financial_extraction import extract_financial_facts_from_report
from src.persistence import (
    build_report_cache_key,
    load_report_artifacts,
    save_report_artifacts,
)
from src.cse_announcements import CSEAnnouncementsClient, CATEGORY_URLS
from src.ui import context_bar, empty_state, inject_global_styles, page_header, render_company_selector
from src.app_state import send_to_analyst_workspace, set_active_symbol




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"

if "report_summary_output" not in st.session_state:
    st.session_state.report_summary_output = ""

if "interim_key_figures_output" not in st.session_state:
    st.session_state.interim_key_figures_output = ""

if "report_compare_output" not in st.session_state:
    st.session_state.report_compare_output = ""

if "latest_report_text" not in st.session_state:
    st.session_state.latest_report_text = ""

if "previous_report_text" not in st.session_state:
    st.session_state.previous_report_text = ""

if "latest_report_label" not in st.session_state:
    st.session_state.latest_report_label = ""

if "previous_report_label" not in st.session_state:
    st.session_state.previous_report_label = ""


@st.cache_data(ttl=3600)
def load_universe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["symbol", "company_name"])

    df = pd.read_csv(p)
    expected = {"symbol", "company_name"}
    if not expected.issubset(df.columns):
        return pd.DataFrame(columns=["symbol", "company_name"])

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["company_name"] = df["company_name"].astype(str).str.strip()
    return df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def normalize_symbol_text(value: str) -> str:
    value = (value or "").strip().upper()
    while ".." in value:
        value = value.replace("..", ".")
    return value


def symbol_root(value: str) -> str:
    return normalize_symbol_text(value).split(".")[0].strip()


def resolve_symbol_from_universe(user_symbol: str, universe_df: pd.DataFrame) -> str:
    user_symbol = normalize_symbol_text(user_symbol)
    if not user_symbol:
        return ""

    if universe_df.empty:
        return user_symbol

    exact = universe_df[universe_df["symbol"] == user_symbol]
    if not exact.empty:
        return exact.iloc[0]["symbol"]

    root = symbol_root(user_symbol)
    root_matches = universe_df[
        universe_df["symbol"].astype(str).str.upper().apply(symbol_root) == root
    ]
    if not root_matches.empty:
        return root_matches.iloc[0]["symbol"]

    return user_symbol


def lookup_company_from_symbol(symbol: str, universe_df: pd.DataFrame) -> str:
    if not symbol or universe_df.empty:
        return ""
    row = universe_df[universe_df["symbol"] == symbol]
    if row.empty:
        return ""
    return row.iloc[0]["company_name"]


page_header(
    "CSE Report Intelligence",
    "Upload or link annual / interim report PDFs, summarize key facts, and compare reporting periods.",
)

universe_df = load_universe(str(UNIVERSE_PATH))

sel_col, type_col = st.columns([3, 1])
with sel_col:
    final_symbol, company_name = render_company_selector(universe_df, key_prefix="rpt_sel")
with type_col:
    report_type = st.selectbox(
        "Report Type",
        ["Interim Report", "Annual Report", "Quarterly Report", "Other"],
    )

context_bar(final_symbol, company_name)

main_tab1, main_tab2, main_tab3 = st.tabs(["Single Report Analysis", "Compare Reports", "CSE Disclosures Feed"])


with main_tab1:
    st.subheader("Single Report Analysis")

    src_col1, src_col2 = st.columns(2)
    latest_report_url = src_col1.text_input(
        "Report PDF URL",
        placeholder="Paste official annual/interim report PDF URL",
    )
    latest_upload = src_col2.file_uploader(
        "Or upload a PDF",
        type=["pdf"],
        key="latest_upload_pdf",
    )

    btn_col1, btn_col2 = st.columns(2)
    analyze_clicked = btn_col1.button("Analyze Report", use_container_width=True)
    interim_clicked = btn_col2.button("Extract Key Interim Figures", use_container_width=True)

    if analyze_clicked or interim_clicked:
        if not latest_report_url.strip() and latest_upload is None:
            st.warning("Provide a report PDF URL or upload a PDF.")
        else:
            try:
                if latest_upload is not None:
                    latest_label = latest_upload.name
                else:
                    latest_label = latest_report_url.strip()

                cache_key = build_report_cache_key(company_name, final_symbol, latest_label)
                cached = load_report_artifacts(cache_key)

                if cached:
                    latest_text = cached.get("text", "")
                    summary = cached.get("summary", "")
                    event = cached.get("event", {})
                    financials = cached.get("financials", {})
                    interim_kf = cached.get("interim_key_figures", "")
                    st.session_state.report_summary_cache_status = "Loaded from AI cache"
                else:
                    if latest_upload is not None:
                        latest_text = extract_pdf_text_from_bytes(latest_upload.getvalue())
                    else:
                        latest_text = extract_pdf_text_from_url(latest_report_url.strip())

                    summary = cached.get("summary", "") if cached else ""
                    event = cached.get("event", {}) if cached else {}
                    financials = cached.get("financials", {}) if cached else {}
                    interim_kf = cached.get("interim_key_figures", "") if cached else ""

                    if interim_clicked and not interim_kf:
                        with st.spinner("Extracting Key Interim Figures with AI analyst engine..."):
                            interim_kf = extract_interim_key_figures(
                                company_name=company_name,
                                ticker=final_symbol,
                                report_text=latest_text,
                            )

                    if analyze_clicked and not summary:
                        with st.spinner("Generating full report intelligence & financial facts..."):
                            summary = summarize_report(
                                company_name=company_name,
                                ticker=final_symbol,
                                report_type=report_type,
                                report_text=latest_text,
                            )
                            event = extract_events_from_report(
                                company_name=company_name,
                                ticker=final_symbol,
                                report_type=report_type,
                                text=latest_text,
                            )
                            financials = extract_financial_facts_from_report(
                                company_name=company_name,
                                ticker=final_symbol,
                                report_type=report_type,
                                text=latest_text,
                            )
                            if not interim_kf:
                                interim_kf = extract_interim_key_figures(
                                    company_name=company_name,
                                    ticker=final_symbol,
                                    report_text=latest_text,
                                )

                    save_report_artifacts(
                        cache_key=cache_key,
                        text=latest_text,
                        summary=summary,
                        event=event,
                        financials=financials,
                        meta={"company_name": company_name, "ticker": final_symbol, "report_label": latest_label},
                        interim_key_figures=interim_kf,
                    )
                    st.session_state.report_summary_cache_status = "Fresh analysis generated"

                st.session_state.latest_report_text = latest_text
                st.session_state.latest_report_label = latest_label
                if summary:
                    st.session_state.report_summary_output = summary
                if interim_kf:
                    st.session_state.interim_key_figures_output = interim_kf
                if event:
                    st.session_state.selected_report_event = event
                if financials:
                    st.session_state.selected_report_financials = financials
            except Exception as e:
                st.error(f"Failed to analyze report: {e}")

    if st.session_state.report_summary_output or st.session_state.interim_key_figures_output:
        st.markdown(f"### {company_name or final_symbol or 'Selected Company'}")
        if final_symbol:
            st.caption(f"Resolved ticker: {final_symbol}")

        if st.session_state.interim_key_figures_output:
            st.markdown("#### Interim Financial Report – Key Figures Analysis")
            st.markdown(st.session_state.interim_key_figures_output)
            st.markdown("---")

        selected_event = st.session_state.get("selected_report_event")
        if selected_event:
            st.markdown("#### Structured Report Signals")
            ei_score = event_importance_score(selected_event)
            cache_status = st.session_state.get("report_summary_cache_status", "Fresh analysis generated")
            st.caption(f"Importance: {ei_score} | Confidence: {selected_event.get('confidence', 'N/A')} | **{cache_status}**")
            
            ecol1, ecol2 = st.columns(2)
            ecol1.write(f"**Report Type**: {selected_event.get('event_type')}")
            ecol2.write(f"**Materiality**: {selected_event.get('materiality_level')}")
            
            pos_sigs = selected_event.get("positive_signals", [])
            if pos_sigs:
                st.write(f"**Positive Signals**: {'; '.join(str(s) for s in pos_sigs)}")
                
            risk_sigs = selected_event.get("risk_signals", [])
            if risk_sigs:
                st.write(f"**Risk Signals**: {'; '.join(str(s) for s in risk_sigs)}")
                
            nums = selected_event.get("key_numbers", [])
            if nums:
                st.write(f"**Key Numbers**: {'; '.join(str(n) for n in nums)}")
                
            unks = selected_event.get("unknowns", [])
            if unks:
                st.write(f"**Unknowns**: {'; '.join(str(u) for u in unks)}")

            st.markdown("---")

        selected_financials = st.session_state.get("selected_report_financials")
        if selected_financials:
            st.markdown("#### Extracted Financial Facts")
            
            fc1, fc2, fc3 = st.columns(3)
            fc1.write(f"**Period**: {selected_financials.get('reporting_period', 'Unknown')}")
            fc2.write(f"**Currency**: {selected_financials.get('currency', 'LKR')}")
            fc3.write(f"**Tone**: {selected_financials.get('management_tone', 'Neutral')}")
            
            metrics = {
                "Revenue": selected_financials.get("revenue"),
                "Rev Growth": selected_financials.get("revenue_growth_pct"),
                "Net Profit": selected_financials.get("net_profit"),
                "EPS": selected_financials.get("eps"),
                "Dividend": selected_financials.get("dividend_per_share")
            }
            
            active_metrics = {k: v for k, v in metrics.items() if v and v != "Unknown"}
            if active_metrics:
                mc = st.columns(len(active_metrics))
                for i, (k, v) in enumerate(active_metrics.items()):
                    mc[i].metric(k, v)
            
            pos = selected_financials.get("positive_signals", [])
            if isinstance(pos, list) and pos:
                st.write(f"**Tech/Qualitative Positives**: {'; '.join(str(p) for p in pos)}")
                
            risk = selected_financials.get("risk_signals", [])
            if isinstance(risk, list) and risk:
                st.write(f"**Risk Shifts**: {'; '.join(str(r) for r in risk)}")
                
            st.markdown("---")

        st.markdown("#### NLP Summary")
        st.write(st.session_state.report_summary_output)

        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Send Summary to Analyst Workspace", use_container_width=True):
            summary_text = st.session_state.report_summary_output
            starter = summary_text[:2500] if summary_text else f"Analyze the latest report for {company_name or final_symbol}."
            send_to_analyst_workspace(
                company_name=company_name,
                ticker=final_symbol,
                analysis_mode="Portfolio Memo",
                query=starter,
            )
        if action_col2.button("Send Key Figures to Analyst Workspace", use_container_width=True):
            kf_text = st.session_state.interim_key_figures_output
            starter = kf_text[:2500] if kf_text else f"Interim Key Figures for {company_name or final_symbol}."
            send_to_analyst_workspace(
                company_name=company_name,
                ticker=final_symbol,
                analysis_mode="Portfolio Memo",
                query=starter,
            )

        with st.expander("Extracted latest report text preview"):
            st.write((st.session_state.latest_report_text or "")[:6000] or "No text extracted.")

with main_tab2:
    st.subheader("Compare Latest vs Previous Report")

    cmp_col1, cmp_col2 = st.columns(2)

    latest_url = cmp_col1.text_input(
        "Latest report PDF URL",
        placeholder="Paste latest report PDF URL",
        key="compare_latest_url",
    )
    latest_file = cmp_col1.file_uploader(
        "Or upload latest report",
        type=["pdf"],
        key="compare_latest_file",
    )

    previous_url = cmp_col2.text_input(
        "Previous report PDF URL",
        placeholder="Paste previous report PDF URL",
        key="compare_previous_url",
    )
    previous_file = cmp_col2.file_uploader(
        "Or upload previous report",
        type=["pdf"],
        key="compare_previous_file",
    )

    if st.button("Compare Reports", use_container_width=True):
        if (not latest_url.strip() and latest_file is None) or (not previous_url.strip() and previous_file is None):
            st.warning("Provide both latest and previous reports.")
        else:
            try:
                if latest_file is not None:
                    latest_text = extract_pdf_text_from_bytes(latest_file.getvalue())
                    latest_label = latest_file.name
                else:
                    latest_text = extract_pdf_text_from_url(latest_url.strip())
                    latest_label = latest_url.strip()

                if previous_file is not None:
                    previous_text = extract_pdf_text_from_bytes(previous_file.getvalue())
                    previous_label = previous_file.name
                else:
                    previous_text = extract_pdf_text_from_url(previous_url.strip())
                    previous_label = previous_url.strip()

                st.session_state.latest_report_text = latest_text
                st.session_state.previous_report_text = previous_text
                st.session_state.latest_report_label = latest_label
                st.session_state.previous_report_label = previous_label

                with st.spinner("Comparing reporting periods..."):
                    comparison = compare_reports(
                        company_name=company_name,
                        ticker=final_symbol,
                        latest_label=latest_label,
                        latest_text=latest_text,
                        previous_label=previous_label,
                        previous_text=previous_text,
                    )

                st.session_state.report_compare_output = comparison
            except Exception as e:
                st.error(f"Failed to compare reports: {e}")

    if st.session_state.report_compare_output:
        st.markdown(f"### {company_name or final_symbol or 'Selected Company'}")
        if final_symbol:
            st.caption(f"Resolved ticker: {final_symbol}")
        st.caption(f"Latest: {st.session_state.latest_report_label}")
        st.caption(f"Previous: {st.session_state.previous_report_label}")
        st.write(st.session_state.report_compare_output)

        with st.expander("Latest report text preview"):
            st.write((st.session_state.latest_report_text or "")[:5000] or "No text extracted.")

        with st.expander("Previous report text preview"):
            st.write((st.session_state.previous_report_text or "")[:5000] or "No text extracted.")

with main_tab3:
    st.subheader("Official CSE Disclosures Feed")

    cat_col1, cat_col2 = st.columns([2, 1])
    selected_cat = cat_col1.selectbox("Category", list(CATEGORY_URLS.keys()), key="doc_intel_cat")
    search_kw = cat_col2.text_input("Filter Keyword", placeholder="e.g. Dividend, Interim, Rights", key="doc_intel_kw")

    try:
        with st.spinner("Fetching disclosures feed..."):
            feed_df = CSEAnnouncementsClient(timeout=10).fetch_announcements(selected_cat)
    except Exception:
        feed_df = pd.DataFrame()

    if isinstance(feed_df, pd.DataFrame) and not feed_df.empty:
        filtered_df = feed_df.copy()
        if search_kw.strip():
            kw_q = search_kw.strip().upper()
            filtered_df = filtered_df[
                filtered_df["announcement_title"].astype(str).str.upper().str.contains(kw_q, na=False)
                | filtered_df["company_name"].astype(str).str.upper().str.contains(kw_q, na=False)
            ]
        
        display_cols = [c for c in ["announcement_date", "company_name", "category", "announcement_title"] if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head(50), use_container_width=True, hide_index=True)
    else:
        empty_state("No Disclosures", "No announcements currently parsed for this category.", "Try selecting 'All' or retry network request.")