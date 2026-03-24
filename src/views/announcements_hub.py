from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.cse_announcements import CSEAnnouncementsClient, CATEGORY_URLS
from src.announcement_intelligence import (
    compare_announcements,
    fetch_announcement_text,
    summarize_announcement_text,
)
from src.event_extraction import (
    extract_events_from_announcement,
    event_importance_score,
)
from src.financial_extraction import extract_financial_facts_from_announcement
from src.persistence import (
    build_announcement_cache_key,
    load_announcement_artifacts,
    save_announcement_artifacts,
)
from src.ui import inject_global_styles, page_header, section_header, status_badge, empty_state, divider_label, chip_row
from src.app_state import send_to_analyst_workspace, send_to_stock_research, set_active_symbol




inject_global_styles()

client = CSEAnnouncementsClient()
BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"

if "selected_announcement_row" not in st.session_state:
    st.session_state.selected_announcement_row = None
if "selected_announcement_text" not in st.session_state:
    st.session_state.selected_announcement_text = ""
if "selected_announcement_summary" not in st.session_state:
    st.session_state.selected_announcement_summary = ""
if "selected_announcement_source" not in st.session_state:
    st.session_state.selected_announcement_source = ""

if "timeline_company" not in st.session_state:
    st.session_state.timeline_company = ""
if "compare_output" not in st.session_state:
    st.session_state.compare_output = ""
if "compare_meta" not in st.session_state:
    st.session_state.compare_meta = {}

page_header(
    "CSE Announcements Hub",
    "Official CSE disclosure feed — search, analyze, and extract intelligence from announcements.",
)


@st.cache_data(ttl=300)
def load_announcements(category: str) -> pd.DataFrame:
    return CSEAnnouncementsClient().fetch_announcements(category)


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


def lookup_symbol_for_company(company_name: str, universe_df: pd.DataFrame) -> str:
    if not company_name or universe_df.empty:
        return ""

    target = company_name.strip().upper()

    exact = universe_df[universe_df["company_name"].str.upper() == target]
    if not exact.empty:
        return exact.iloc[0]["symbol"]

    root = (
        target.replace(" PLC", "")
        .replace(" LIMITED", "")
        .replace(" LTD", "")
        .replace(" THE ", " ")
        .strip()
    )

    broad = universe_df[
        universe_df["company_name"].str.upper().str.contains(root, na=False)
    ]
    if not broad.empty:
        return broad.iloc[0]["symbol"]

    return ""


def score_importance(title: str, category: str) -> tuple[str, int]:
    text = f"{title} {category}".upper()

    high_keywords = [
        "RIGHTS ISSUE",
        "ACQUISITION",
        "MERGER",
        "TAKEOVER",
        "DIVIDEND",
        "INTERIM FINANCIAL STATEMENTS",
        "ANNUAL REPORT",
        "PROFIT WARNING",
        "BOARD MEETING",
        "SHARE SPLIT",
        "DELIST",
        "BONUS ISSUE",
        "MATERIAL",
        "CAPITAL",
    ]

    medium_keywords = [
        "DIRECTOR",
        "CEO",
        "CFO",
        "RESIGN",
        "APPOINT",
        "DISCLOSURE",
        "RELATED PARTY",
        "TRANSACTION",
        "LITIGATION",
        "REGULATORY",
    ]

    if any(k in text for k in high_keywords):
        return "High", 3
    if any(k in text for k in medium_keywords):
        return "Medium", 2
    return "Routine", 1


def badge_color(level: str) -> str:
    if level == "High":
        return "Priority"
    if level == "Medium":
        return "Watch"
    return "Routine"


def analyze_selected_announcement(row_dict: dict):
    company_name = str(row_dict.get("company_name", "")).strip()
    ticker = str(row_dict.get("mapped_ticker", "")).strip()
    title = str(row_dict.get("announcement_title", "")).strip()
    category = str(row_dict.get("category", "")).strip()
    detail_url = str(row_dict.get("detail_url", "")).strip()
    pdf_url = str(row_dict.get("pdf_url", "")).strip()
    source_page = str(row_dict.get("source_page", "")).strip()

    if ticker or company_name:
        set_active_symbol(ticker, company_name)

    source_to_read = pdf_url or detail_url or source_page
    cache_key = build_announcement_cache_key(company_name, title, source_to_read)
    cached = load_announcement_artifacts(cache_key)
    
    if cached:
        text = cached.get("text", "")
        summary = cached.get("summary", "")
        event = cached.get("event", {})
        financials = cached.get("financials", {})
        st.session_state.selected_announcement_cache_status = "Loaded from AI cache"
    else:
        text = fetch_announcement_text(source_to_read)
        summary = summarize_announcement_text(
            company_name=company_name,
            title=title,
            category=category,
            text=text,
        )
        event = extract_events_from_announcement(
            company_name=company_name,
            ticker=ticker,
            title=title,
            category=category,
            text=text,
        )
        financials = extract_financial_facts_from_announcement(
            company_name=company_name,
            ticker=ticker,
            title=title,
            category=category,
            text=text,
        )
        save_announcement_artifacts(
            cache_key=cache_key,
            text=text,
            summary=summary,
            event=event,
            financials=financials,
            meta={"company_name": company_name, "ticker": ticker, "title": title, "source_url": source_to_read}
        )
        st.session_state.selected_announcement_cache_status = "Fresh analysis generated"

    st.session_state.selected_announcement_row = row_dict
    st.session_state.selected_announcement_text = text
    st.session_state.selected_announcement_summary = summary
    st.session_state.selected_announcement_event = event
    st.session_state.selected_announcement_financials = financials
    st.session_state.selected_announcement_source = source_to_read
    st.session_state.timeline_company = company_name


def compare_latest_previous(company_df: pd.DataFrame):
    if len(company_df) < 2:
        st.session_state.compare_output = "Not enough announcements to compare."
        st.session_state.compare_meta = {}
        return

    latest = company_df.iloc[0].to_dict()
    previous = company_df.iloc[1].to_dict()

    latest_source = latest.get("pdf_url") or latest.get("detail_url") or latest.get("source_page")
    previous_source = previous.get("pdf_url") or previous.get("detail_url") or previous.get("source_page")

    latest_text = fetch_announcement_text(str(latest_source))
    previous_text = fetch_announcement_text(str(previous_source))

    company_name = str(latest.get("company_name", "")).strip()
    latest_title = str(latest.get("announcement_title", "")).strip()
    previous_title = str(previous.get("announcement_title", "")).strip()

    comparison = compare_announcements(
        company_name=company_name,
        latest_title=latest_title,
        latest_text=latest_text,
        previous_title=previous_title,
        previous_text=previous_text,
    )

    st.session_state.compare_output = comparison
    st.session_state.compare_meta = {
        "company_name": company_name,
        "latest_title": latest_title,
        "previous_title": previous_title,
        "latest_date": latest.get("announcement_date", ""),
        "previous_date": previous.get("announcement_date", ""),
    }


universe_df = load_universe(str(UNIVERSE_PATH))

filter_top1, filter_top2, filter_top3 = st.columns([1, 2, 1])

selected_category = filter_top1.selectbox(
    "Category",
    list(CATEGORY_URLS.keys()),
)

keyword = filter_top2.text_input(
    "Company / keyword search",
    placeholder="e.g. John Keells, dividend, rights issue, director",
)

days_back = filter_top3.selectbox(
    "Date Window",
    ["All", "7D", "30D", "90D", "180D"],
    index=1,
)

with st.spinner("Loading official CSE announcements..."):
    announcements_df = load_announcements(selected_category)

if announcements_df.empty:
    st.warning("No announcement rows were parsed from the current CSE page.")

    debug_dir = Path("data/debug")
    text_file = debug_dir / "last_cse_announcements_visible_text.txt"
    html_file = debug_dir / "last_cse_announcements_rendered.html"

    with st.expander("Debug files preview"):
        st.write(f"Visible text file exists: {text_file.exists()}")
        st.write(f"Rendered HTML file exists: {html_file.exists()}")

        if text_file.exists():
            preview = text_file.read_text(encoding="utf-8", errors="ignore")
            st.text(preview[:4000])

    st.stop()

work = announcements_df.copy()

work["mapped_ticker"] = work["company_name"].apply(lambda x: lookup_symbol_for_company(str(x), universe_df))
work["has_pdf"] = work["pdf_url"].astype(str).str.len() > 0
work["importance_label"], work["importance_score"] = zip(
    *work.apply(
        lambda row: score_importance(
            str(row.get("announcement_title", "")),
            str(row.get("category", "")),
        ),
        axis=1,
    )
)

if "announcement_date_parsed" not in work.columns:
    work["announcement_date_parsed"] = pd.to_datetime(
        work["announcement_date"],
        errors="coerce",
        dayfirst=True,
    )

if keyword.strip():
    q = keyword.strip().upper()
    work = work[
        work["company_name"].str.upper().str.contains(q, na=False)
        | work["announcement_title"].str.upper().str.contains(q, na=False)
        | work["category"].str.upper().str.contains(q, na=False)
        | work["mapped_ticker"].str.upper().str.contains(q, na=False)
    ]

toggle_col1, toggle_col2, toggle_col3 = st.columns(3)
only_pdfs = toggle_col1.toggle("Only PDF-backed announcements", value=False)
only_mapped = toggle_col2.toggle("Only mapped-to-ticker items", value=False)
high_priority_only = toggle_col3.toggle("Only high-priority items", value=False)

if only_pdfs:
    work = work[work["has_pdf"]]

if only_mapped:
    work = work[work["mapped_ticker"].astype(str).str.len() > 0]

if high_priority_only:
    work = work[work["importance_label"] == "High"]

if days_back != "All":
    days = int(days_back.replace("D", ""))
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    work = work[work["announcement_date_parsed"] >= cutoff]

work = work.sort_values(
    by=["importance_score", "announcement_date_parsed"],
    ascending=[False, False],
    na_position="last",
).reset_index(drop=True)

top_metrics = st.columns(4)
top_metrics[0].metric("Announcements", len(work))
top_metrics[1].metric("Companies", work["company_name"].nunique())
top_metrics[2].metric("PDFs", int(work["has_pdf"].sum()))
top_metrics[3].metric("High Priority", int((work["importance_label"] == "High").sum()))

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ["Feed", "Company Timeline", "Table View", "Debug"]
)

with main_tab1:
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("Announcement Feed")

        if work.empty:
            st.info("No announcements match the current filters.")
        else:
            for idx, row in work.head(40).iterrows():
                company_name = str(row.get("company_name", "")).strip()
                title = str(row.get("announcement_title", "")).strip()
                date_text = str(row.get("announcement_date", "")).strip()
                category = str(row.get("category", "")).strip()
                detail_url = str(row.get("detail_url", "")).strip()
                pdf_url = str(row.get("pdf_url", "")).strip()
                source_page = str(row.get("source_page", "")).strip()
                ticker = str(row.get("mapped_ticker", "")).strip()
                importance = str(row.get("importance_label", "Routine")).strip()

                row_dict = row.to_dict()

                with st.container(border=True):
                    st.markdown(f"**{company_name or 'Unknown Company'}**")
                    badge_html = status_badge(importance, importance.lower())
                    st.markdown(
                        f"<div style='margin-bottom:4px;'>{badge_html} <span style='opacity:0.6; font-size:0.85em;'>{date_text} • {category}</span></div>",
                        unsafe_allow_html=True
                    )
                    st.write(title or "No title parsed.")

                    meta_col1, meta_col2 = st.columns(2)
                    meta_col1.caption(f"Ticker: {ticker or 'Not mapped'}")
                    meta_col2.caption(f"PDF: {'Yes' if pdf_url else 'No'}")

                    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

                    if detail_url:
                        btn_col1.link_button("Open", detail_url, use_container_width=True)
                    elif source_page:
                        btn_col1.link_button("Open", source_page, use_container_width=True)

                    if pdf_url:
                        btn_col2.link_button("PDF", pdf_url, use_container_width=True)
                    else:
                        btn_col2.write("")

                    if btn_col3.button(
                        "Analyze",
                        key=f"analyze_announcement_{idx}",
                        use_container_width=True,
                    ):
                        with st.spinner("Reading and analyzing announcement..."):
                            analyze_selected_announcement(row_dict)
                        st.rerun()

                    if btn_col4.button(
                        "Timeline",
                        key=f"timeline_announcement_{idx}",
                        use_container_width=True,
                    ):
                        st.session_state.timeline_company = company_name
                        st.rerun()

                    if btn_col5.button(
                        "Send to Analyst",
                        key=f"send_announcement_{idx}",
                        use_container_width=True,
                    ):
                        send_to_analyst_workspace(
                            company_name=company_name,
                            ticker=ticker,
                            query=f"Analyze this CSE announcement for investment implications, catalysts, risks, and what changed: {title}",
                        )

    with right_col:
        st.subheader("Announcement Intelligence")

        selected_row = st.session_state.get("selected_announcement_row")
        selected_summary = st.session_state.get("selected_announcement_summary")
        selected_event = st.session_state.get("selected_announcement_event")
        selected_text = st.session_state.get("selected_announcement_text")
        selected_source = st.session_state.get("selected_announcement_source")

        if selected_row and selected_summary:
            st.markdown(f"**{selected_row.get('company_name', 'Unknown Company')}**")
            st.caption(
                f"{selected_row.get('announcement_date', '')} • {selected_row.get('category', '')}"
            )
            st.write(selected_row.get("announcement_title", ""))

            if selected_source:
                st.link_button(
                    "Open analyzed source",
                    selected_source,
                    use_container_width=True,
                )

            st.markdown("---")

            if selected_event:
                st.markdown("### Structured Event")
                
                ei_score = event_importance_score(selected_event)
                cache_status = st.session_state.get("selected_announcement_cache_status", "Fresh analysis generated")
                st.caption(f"Importance: {ei_score} | Confidence: {selected_event.get('confidence', 'N/A')} | **{cache_status}**")
                
                ecol1, ecol2 = st.columns(2)
                ecol1.write(f"**Event Type**: {selected_event.get('event_type')}")
                ecol2.write(f"**Materiality**: {selected_event.get('materiality_level')}")
                
                if selected_event.get("effective_date"):
                    st.write(f"**Effective Date**: {selected_event.get('effective_date')}")
                    
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

            selected_financials = st.session_state.get("selected_announcement_financials")
            if selected_financials:
                st.markdown("### Financial Signals")
                f_col1, f_col2 = st.columns(2)
                f_col1.write(f"**Period**: {selected_financials.get('reporting_period', 'Unknown')}")
                f_col2.write(f"**Management Tone**: {selected_financials.get('management_tone', 'Neutral')}")
                
                sig_lines = []
                for lab, k in [("Dividend/Payout", "payout_signal"), ("Guidance", "guidance_signal"), ("Margin", "margin_signal"), ("Liquidity", "liquidity_signal"), ("Leverage", "leverage_signal")]:
                    val = selected_financials.get(k)
                    if val and val != "Unknown":
                        sig_lines.append(f"- **{lab}**: {val}")
                        
                if sig_lines:
                    for line in sig_lines:
                        st.write(line)
                else:
                    st.caption("No overarching financial signals isolated.")
                
                key_nums = selected_financials.get("key_numbers", [])
                if isinstance(key_nums, list) and key_nums:
                    with st.expander("Key Numbers Detected"):
                        for num in key_nums:
                            st.caption(f"- {num}")
                            
                st.markdown("---")

            st.markdown("### NLP Summary")
            st.write(selected_summary)

            with st.expander("Extracted source text preview"):
                st.write((selected_text or "")[:5000] or "No text extracted.")
        else:
            st.info("Click **Analyze** on any announcement to generate an AI summary here.")

with main_tab2:
    st.subheader("Company Disclosure Timeline")

    company_options = sorted(work["company_name"].dropna().astype(str).unique().tolist())
    default_index = 0
    if st.session_state.timeline_company and st.session_state.timeline_company in company_options:
        default_index = company_options.index(st.session_state.timeline_company)

    selected_company = st.selectbox(
        "Select company",
        company_options,
        index=default_index if company_options else None,
    ) if company_options else ""

    if selected_company:
        company_df = work[work["company_name"] == selected_company].copy()
        company_df = company_df.sort_values(
            by="announcement_date_parsed",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        ticker = lookup_symbol_for_company(selected_company, universe_df)
        if ticker or selected_company:
            set_active_symbol(ticker, selected_company)
        top_col1, top_col2, top_col3 = st.columns(3)
        top_col1.metric("Disclosures", len(company_df))
        top_col2.metric("Mapped Ticker", ticker or "N/A")
        top_col3.metric("High Priority", int((company_df["importance_label"] == "High").sum()))

        latest_box, previous_box = st.columns(2)

        if len(company_df) >= 1:
            latest = company_df.iloc[0]
            with latest_box:
                st.markdown("**Latest Disclosure**")
                st.caption(f"{latest.get('announcement_date', '')} • {latest.get('category', '')}")
                st.write(latest.get("announcement_title", ""))

        if len(company_df) >= 2:
            previous = company_df.iloc[1]
            with previous_box:
                st.markdown("**Previous Disclosure**")
                st.caption(f"{previous.get('announcement_date', '')} • {previous.get('category', '')}")
                st.write(previous.get("announcement_title", ""))

        action_col1, action_col2, action_col3 = st.columns(3)
        if action_col1.button("Compare Latest vs Previous", use_container_width=True):
            with st.spinner("Comparing latest and previous disclosures..."):
                compare_latest_previous(company_df)
            st.rerun()

        if action_col2.button("Send Company to Analyst Workspace", use_container_width=True):
            latest_title = company_df.iloc[0]["announcement_title"] if len(company_df) else selected_company
            send_to_analyst_workspace(
                company_name=selected_company,
                ticker=ticker,
                query=f"Analyze this CSE announcement for investment implications, catalysts, risks, and what changed: {latest_title}",
            )

        if action_col3.button("Open Stock Research", use_container_width=True):
            send_to_stock_research(ticker, selected_company)

        if st.session_state.compare_output and st.session_state.compare_meta.get("company_name") == selected_company:
            st.markdown("### AI Comparison")
            meta = st.session_state.compare_meta
            st.caption(
                f"Latest: {meta.get('latest_date', '')} — {meta.get('latest_title', '')}"
            )
            st.caption(
                f"Previous: {meta.get('previous_date', '')} — {meta.get('previous_title', '')}"
            )
            st.write(st.session_state.compare_output)

        st.markdown("### Timeline Table")
        show_cols = [
            c for c in [
                "announcement_date",
                "announcement_title",
                "category",
                "importance_label",
                "mapped_ticker",
                "detail_url",
                "pdf_url",
            ]
            if c in company_df.columns
        ]
        st.dataframe(company_df[show_cols], use_container_width=True, hide_index=True)

with main_tab3:
    st.subheader("Announcement Table")

    show_cols = [
        c for c in [
            "announcement_date",
            "company_name",
            "mapped_ticker",
            "announcement_title",
            "category",
            "importance_label",
            "detail_url",
            "pdf_url",
        ]
        if c in work.columns
    ]

    st.dataframe(
        work[show_cols],
        use_container_width=True,
        hide_index=True,
    )

with main_tab4:
    st.subheader("Debug Preview")
    st.write(f"Universe rows: {len(universe_df)}")
    st.write(f"Announcement rows loaded: {len(announcements_df)}")
    st.write(f"Announcement rows after filters: {len(work)}")
    st.dataframe(work.head(10), use_container_width=True, hide_index=True)