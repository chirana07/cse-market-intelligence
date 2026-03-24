from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.yahoo_prices import YahooCSEClient
from src.cse_announcements import CSEAnnouncementsClient
from src.announcement_intelligence import (
    fetch_announcement_text,
    summarize_announcement_text,
)
from src.report_intelligence import (
    extract_pdf_text_from_bytes,
    extract_pdf_text_from_url,
    summarize_report,
)
from src.stock_research_intelligence import generate_stock_ai_view
from src.event_extraction import (
    extract_events_from_announcement,
    extract_events_from_report,
    event_importance_score,
    event_to_markdown,
)
from src.financial_extraction import (
    extract_financial_facts_from_announcement,
    extract_financial_facts_from_report,
    financial_fact_to_markdown,
)
from src.persistence import (
    build_announcement_cache_key,
    load_announcement_artifacts,
    save_announcement_artifacts,
    build_report_cache_key,
    load_report_artifacts,
    save_report_artifacts,
    build_stock_cache_key,
    load_stock_ai_view,
    save_stock_ai_view,
    _safe_hash,
)
from src.ui import inject_global_styles, page_header, section_header, status_badge, empty_state, divider_label, chip_row
from src.app_state import send_to_analyst_workspace, set_active_symbol




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"

yahoo_client = YahooCSEClient(universe_path=UNIVERSE_PATH)
ann_client = CSEAnnouncementsClient()

# ---------------------------
# Session state
# ---------------------------
if "stock_research_announcement_summary" not in st.session_state:
    st.session_state.stock_research_announcement_summary = ""

if "stock_research_announcement_text" not in st.session_state:
    st.session_state.stock_research_announcement_text = ""

if "stock_research_report_summary" not in st.session_state:
    st.session_state.stock_research_report_summary = ""

if "stock_research_report_text" not in st.session_state:
    st.session_state.stock_research_report_text = ""

if "stock_research_announcement_financials" not in st.session_state:
    st.session_state.stock_research_announcement_financials = {}
if "stock_research_report_financials" not in st.session_state:
    st.session_state.stock_research_report_financials = {}
if "stock_research_ai_view" not in st.session_state:
    st.session_state.stock_research_ai_view = ""

pending_symbol = st.session_state.pop("pending_stock_research_symbol", None)
if pending_symbol:
    st.session_state.stock_research_prefill_symbol = pending_symbol

if "stock_research_prefill_symbol" not in st.session_state:
    st.session_state.stock_research_prefill_symbol = ""


# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    return YahooCSEClient(universe_path=path).load_universe()


@st.cache_data(ttl=900)
def load_history_cached(symbol: str, universe_path: str, period: str, interval: str) -> pd.DataFrame:
    local_client = YahooCSEClient(universe_path=universe_path)
    return local_client.get_history(symbol, period=period, interval=interval)


@st.cache_data(ttl=300)
def load_announcements_cached() -> pd.DataFrame:
    return CSEAnnouncementsClient().fetch_announcements("All")


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value, decimals=2):
    number = _num(value)
    if number is None:
        return "N/A"
    return f"{number:,.{decimals}f}"


def _fmt_delta(change, pct):
    c = _num(change)
    p = _num(pct)
    if c is None and p is None:
        return "N/A"
    if c is None:
        return f"{p:.2f}%"
    if p is None:
        return f"{c:,.2f}"
    return f"{c:,.2f} ({p:.2f}%)"


def _standardize_history(hist: pd.DataFrame) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = hist.copy()
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    return df


def _pct_change(current, previous):
    current = _num(current)
    previous = _num(previous)
    if current is None or previous in (None, 0):
        return None
    return ((current - previous) / previous) * 100


def _return_from_days(df: pd.DataFrame, days: int):
    if df.empty or "Close" not in df.columns:
        return None
    current = _num(df["Close"].iloc[-1])
    if current is None:
        return None

    cutoff = df["Date"].iloc[-1] - pd.Timedelta(days=days)
    earlier = df[df["Date"] <= cutoff]
    if earlier.empty:
        return None

    previous = _num(earlier["Close"].iloc[-1])
    return _pct_change(current, previous)


def _ytd_return(df: pd.DataFrame):
    if df.empty:
        return None
    year = df["Date"].iloc[-1].year
    ytd_rows = df[df["Date"].dt.year == year]
    if ytd_rows.empty:
        return None
    start_close = _num(ytd_rows["Close"].iloc[0])
    end_close = _num(ytd_rows["Close"].iloc[-1])
    return _pct_change(end_close, start_close)


def _build_price_chart(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA50"], mode="lines", name="SMA 50"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def _build_volume_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume"))
    fig.update_layout(
        title="Volume Trend",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig



def map_company_to_ticker(company_name: str, universe_df: pd.DataFrame) -> str:
    if not company_name or universe_df.empty:
        return ""

    target = str(company_name).strip().upper()

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


# ---------------------------
# Load universe + announcements
# ---------------------------
universe_df = load_universe_cached(str(UNIVERSE_PATH))
all_announcements = load_announcements_cached()

page_header(
    "CSE Stock Research",
    "Unified intelligence page — price history, disclosures, financials, and AI-synthesized insights.",
)

search_col1, search_col2 = st.columns([2, 1])

search_text = search_col1.text_input(
    "Search company or symbol",
    placeholder="e.g. John Keells, JKH, COMB",
)

matches = universe_df.copy()
if search_text.strip():
    q = search_text.strip().upper()
    matches = matches[
        matches["symbol"].str.contains(q, na=False)
        | matches["company_name"].str.upper().str.contains(q, na=False)
    ]

option_map = {
    f"{row['company_name']} ({row['symbol']})": row["symbol"]
    for _, row in matches.head(100).iterrows()
}

selected_label = search_col1.selectbox(
    "Pick from company universe",
    options=[""] + list(option_map.keys()),
    index=0,
)

manual_symbol = search_col2.text_input(
    "Or type symbol / alias",
    key="stock_research_prefill_symbol",
    placeholder="e.g. JKH, JKH.N, JKH.N0000",
)

typed_symbol = yahoo_client.normalize_symbol_text(manual_symbol)
selected_symbol = option_map.get(selected_label, "")
final_symbol = (
    yahoo_client.resolve_symbol_from_universe(typed_symbol, universe_df)
    if typed_symbol
    else selected_symbol
)

if not final_symbol:
    st.info("Select a company to open the stock research page.")
    st.stop()

quote = yahoo_client.get_quote(final_symbol)
hist_raw = load_history_cached(final_symbol, str(UNIVERSE_PATH), "1y", "1d")
hist = _standardize_history(hist_raw)

company_name = quote.company_name or yahoo_client.get_company_name(final_symbol, universe_df) or final_symbol

if final_symbol or company_name:
    set_active_symbol(final_symbol, company_name)

company_announcements = all_announcements.copy()
if not company_announcements.empty:
    company_announcements["mapped_ticker"] = company_announcements["company_name"].apply(
        lambda x: map_company_to_ticker(str(x), universe_df)
    )
    company_announcements = company_announcements[
        (company_announcements["mapped_ticker"] == final_symbol)
        | (company_announcements["company_name"].str.upper().str.contains(company_name.upper(), na=False))
    ].copy()

    if "announcement_date_parsed" in company_announcements.columns:
        company_announcements = company_announcements.sort_values(
            by="announcement_date_parsed",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

st.markdown(f"## {company_name}")
header_col1, header_col2, header_col3 = st.columns([1, 1, 1])
header_col1.caption(f"Ticker: {final_symbol}")
header_col2.caption(f"Yahoo symbol: {quote.yahoo_symbol}")
header_col3.button(
    "Send to Analyst Workspace",
    on_click=send_to_analyst_workspace,
    args=(
        company_name,
        final_symbol,
        "Portfolio Memo",
        f"Build a research memo for {company_name or final_symbol}, including market view, latest disclosures, and key risks.",
    ),
    use_container_width=True,
)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Last Traded Price", _fmt_num(quote.last_traded_price), _fmt_delta(quote.change, quote.change_pct))
metric_col2.metric("Open", _fmt_num(quote.open_price))
metric_col3.metric("High / Low", f"{_fmt_num(quote.high)} / {_fmt_num(quote.low)}")
metric_col4.metric("Volume", _fmt_num(quote.volume, decimals=0))

metric_col5, metric_col6, metric_col7, metric_col8 = st.columns(4)
metric_col5.metric("Previous Close", _fmt_num(quote.previous_close))
metric_col6.metric("Market Cap", _fmt_num(quote.market_cap))
if not hist.empty:
    year_slice = hist.tail(min(len(hist), 252))
    metric_col7.metric("52W High", _fmt_num(year_slice["High"].max()))
    metric_col8.metric("52W Low", _fmt_num(year_slice["Low"].min()))
else:
    metric_col7.metric("52W High", "N/A")
    metric_col8.metric("52W Low", "N/A")

tabs = st.tabs(["Overview", "Market", "Announcements", "Reports", "AI View"])

with tabs[0]:
    st.subheader("Overview")

    if not hist.empty:
        ret_col1, ret_col2, ret_col3, ret_col4, ret_col5 = st.columns(5)
        ret_col1.metric("1W Return", _fmt_delta(None, _return_from_days(hist, 7)))
        ret_col2.metric("1M Return", _fmt_delta(None, _return_from_days(hist, 30)))
        ret_col3.metric("3M Return", _fmt_delta(None, _return_from_days(hist, 90)))
        ret_col4.metric("6M Return", _fmt_delta(None, _return_from_days(hist, 180)))
        ret_col5.metric("YTD Return", _fmt_delta(None, _ytd_return(hist)))

        st.plotly_chart(
            _build_price_chart(hist, f"{company_name} Price History"),
            use_container_width=True,
            key=f"overview_price_chart_{final_symbol}",
        )

    ov_col1, ov_col2 = st.columns(2)

    with ov_col1:
        st.markdown("### Latest Disclosure")
        if company_announcements.empty:
            st.info("No company-specific announcements found.")
        else:
            latest_ann = company_announcements.iloc[0]
            st.caption(f"{latest_ann.get('announcement_date', '')} • {latest_ann.get('category', '')}")
            st.write(latest_ann.get("announcement_title", ""))
            if latest_ann.get("pdf_url"):
                st.link_button("Open PDF", latest_ann.get("pdf_url"), use_container_width=True)
            elif latest_ann.get("detail_url"):
                st.link_button("Open Announcement", latest_ann.get("detail_url"), use_container_width=True)

    with ov_col2:
        st.markdown("### Latest Report Intelligence")
        if st.session_state.stock_research_report_summary:
            st.write(st.session_state.stock_research_report_summary[:1800])
        else:
            st.info("No report summary generated yet. Use the Reports tab.")

with tabs[1]:
    st.subheader("Market")

    if hist.empty:
        st.warning("No historical market data returned.")
    else:
        st.plotly_chart(
            _build_price_chart(hist, f"{company_name} Price History"),
            use_container_width=True,
            key=f"market_price_chart_{final_symbol}",
        )
        st.plotly_chart(
            _build_volume_chart(hist),
            use_container_width=True,
            key=f"market_volume_chart_{final_symbol}",
        )

        with st.expander("Show historical market data"):
            st.dataframe(hist, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Announcements")

    if company_announcements.empty:
        st.info("No company announcements found in the currently loaded CSE feed.")
    else:
        for idx, row in company_announcements.head(15).iterrows():
            title = str(row.get("announcement_title", "")).strip()
            category = str(row.get("category", "")).strip()
            date_text = str(row.get("announcement_date", "")).strip()
            detail_url = str(row.get("detail_url", "")).strip()
            pdf_url = str(row.get("pdf_url", "")).strip()
            source_to_read = pdf_url or detail_url or str(row.get("source_page", "")).strip()

            with st.container(border=True):
                st.caption(f"{date_text} • {category}")
                st.write(title)

                bcol1, bcol2, bcol3 = st.columns(3)
                if detail_url:
                    bcol1.link_button("Open", detail_url, use_container_width=True)
                elif pdf_url:
                    bcol1.link_button("Open PDF", pdf_url, use_container_width=True)

                if bcol2.button(f"Analyze #{idx}", use_container_width=True):
                    with st.spinner("Analyzing announcement..."):
                        cache_key = build_announcement_cache_key(company_name, title, source_to_read)
                        cached = load_announcement_artifacts(cache_key)

                        if cached:
                            announcement_text = cached.get("text", "")
                            announcement_summary = cached.get("summary", "")
                            event = cached.get("event", {})
                            financials = cached.get("financials", {})
                            st.session_state.stock_research_announcement_cache_status = "Loaded from AI cache"
                        else:
                            announcement_text = fetch_announcement_text(source_to_read)
                            announcement_summary = summarize_announcement_text(
                                company_name=company_name,
                                title=title,
                                category=category,
                                text=announcement_text,
                            )
                            event = extract_events_from_announcement(
                                company_name=company_name,
                                ticker=ticker,
                                title=title,
                                category=category,
                                text=announcement_text,
                            )
                            financials = extract_financial_facts_from_announcement(
                                company_name=company_name,
                                ticker=ticker,
                                title=title,
                                category=category,
                                text=announcement_text,
                            )
                            save_announcement_artifacts(
                                cache_key=cache_key,
                                text=announcement_text,
                                summary=announcement_summary,
                                event=event,
                                financials=financials,
                                meta={"company_name": company_name, "ticker": ticker, "title": title, "source_url": source_to_read}
                            )
                            st.session_state.stock_research_announcement_cache_status = "Fresh analysis generated"

                        st.session_state.stock_research_announcement_text = announcement_text
                        st.session_state.stock_research_announcement_summary = announcement_summary
                        st.session_state.stock_research_selected_announcement_event = event
                        st.session_state.stock_research_announcement_financials = financials
                    st.rerun()

        if st.session_state.stock_research_announcement_summary:
            st.markdown("### Announcement Intelligence")
            
            selected_event = st.session_state.get("stock_research_selected_announcement_event")
            if selected_event:
                st.markdown("#### Structured Event")
                ei_score = event_importance_score(selected_event)
                cache_status = st.session_state.get("stock_research_announcement_cache_status", "Fresh analysis generated")
                st.caption(f"Importance: {ei_score} | Confidence: {selected_event.get('confidence', 'N/A')} | **{cache_status}**")
                
                ecol1, ecol2 = st.columns(2)
                ecol1.write(f"**Event Type**: {selected_event.get('event_type')}")
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

            selected_financials = st.session_state.get("stock_research_announcement_financials")
            if selected_financials:
                st.markdown("#### Financial Signals")
                f_col1, f_col2 = st.columns(2)
                f_col1.write(f"**Period**: {selected_financials.get('reporting_period', 'Unknown')}")
                f_col2.write(f"**Tone**: {selected_financials.get('management_tone', 'Neutral')}")
                sig_lines = []
                for lab, k in [("Dividend/Payout", "payout_signal"), ("Guidance", "guidance_signal"), ("Margin", "margin_signal"), ("Liquidity", "liquidity_signal"), ("Leverage", "leverage_signal")]:
                    val = selected_financials.get(k)
                    if val and val != "Unknown":
                        sig_lines.append(f"- **{lab}**: {val}")
                if sig_lines:
                    for line in sig_lines:
                        st.write(line)
                st.markdown("---")

            st.markdown("#### NLP Summary")
            st.write(st.session_state.stock_research_announcement_summary)

            with st.expander("Announcement text preview"):
                st.write((st.session_state.stock_research_announcement_text or "")[:6000] or "No text extracted.")

with tabs[3]:
    st.subheader("Reports")

    rep_col1, rep_col2 = st.columns(2)
    report_url = rep_col1.text_input(
        "Report PDF URL",
        placeholder="Paste annual/interim report PDF URL",
    )
    report_upload = rep_col2.file_uploader(
        "Or upload report PDF",
        type=["pdf"],
        key="stock_research_report_upload",
    )

    if st.button("Analyze Report", use_container_width=True):
        if not report_url.strip() and report_upload is None:
            st.warning("Provide a report PDF URL or upload a PDF.")
        else:
            try:
                if report_upload is not None:
                    report_label = report_upload.name
                else:
                    report_label = report_url.strip()

                cache_key = build_report_cache_key(company_name, final_symbol, report_label)
                cached = load_report_artifacts(cache_key)

                if cached:
                    report_text = cached.get("text", "")
                    report_summary = cached.get("summary", "")
                    report_event = cached.get("event", {})
                    report_financials = cached.get("financials", {})
                    st.session_state.stock_research_report_cache_status = "Loaded from AI cache"
                else:
                    if report_upload is not None:
                        report_text = extract_pdf_text_from_bytes(report_upload.getvalue())
                    else:
                        report_text = extract_pdf_text_from_url(report_url.strip())

                    with st.spinner("Generating report intelligence..."):
                        report_summary = summarize_report(
                            company_name=company_name,
                            ticker=final_symbol,
                            report_type="Company Report",
                            report_text=report_text,
                        )
                        report_event = extract_events_from_report(
                            company_name=company_name,
                            ticker=final_symbol,
                            report_type="Company Report",
                            text=report_text,
                        )
                        report_financials = extract_financial_facts_from_report(
                            company_name=company_name,
                            ticker=final_symbol,
                            report_type="Company Report",
                            text=report_text,
                        )
                        
                    save_report_artifacts(
                        cache_key=cache_key,
                        text=report_text,
                        summary=report_summary,
                        event=report_event,
                        financials=report_financials,
                        meta={"company_name": company_name, "ticker": final_symbol, "report_label": report_label}
                    )
                    st.session_state.stock_research_report_cache_status = "Fresh analysis generated"

                st.session_state.stock_research_report_text = report_text
                st.session_state.stock_research_report_summary = report_summary
                st.session_state.stock_research_selected_report_event = report_event
                st.session_state.stock_research_report_financials = report_financials
            except Exception as e:
                st.error(f"Failed to analyze report: {e}")

    if st.session_state.stock_research_report_summary:
        st.markdown("### Report Intelligence")
        
        selected_event = st.session_state.get("stock_research_selected_report_event")
        if selected_event:
            st.markdown("#### Structured Report Signals")
            ei_score = event_importance_score(selected_event)
            cache_status = st.session_state.get("stock_research_report_cache_status", "Fresh analysis generated")
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

        selected_financials = st.session_state.get("stock_research_report_financials")
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
                    
            st.markdown("---")

        st.markdown("#### NLP Summary")
        st.write(st.session_state.stock_research_report_summary)

        with st.expander("Report text preview"):
            st.write((st.session_state.stock_research_report_text or "")[:6000] or "No text extracted.")

with tabs[4]:
    st.subheader("AI View")

    if st.button("Generate AI Stock View", use_container_width=True):
        quote_snapshot = f"""
Last Price: {quote.last_traded_price}
Change: {quote.change}
Change %: {quote.change_pct}
Open: {quote.open_price}
High: {quote.high}
Low: {quote.low}
Previous Close: {quote.previous_close}
Volume: {quote.volume}
Market Cap: {quote.market_cap}
""".strip()

        market_stats = ""
        if not hist.empty:
            market_stats = f"""
1W Return: {_return_from_days(hist, 7)}
1M Return: {_return_from_days(hist, 30)}
3M Return: {_return_from_days(hist, 90)}
6M Return: {_return_from_days(hist, 180)}
YTD Return: {_ytd_return(hist)}
52W High: {hist['High'].max()}
52W Low: {hist['Low'].min()}
""".strip()

        announcement_md = st.session_state.stock_research_announcement_summary or ""
        announcement_event = st.session_state.get("stock_research_selected_announcement_event")
        announcement_fin = st.session_state.get("stock_research_announcement_financials")
        if announcement_event:
            announcement_md += f"\\n\\nStructured Event Signals:\\n{event_to_markdown(announcement_event)}"
        if announcement_fin:
            announcement_md += f"\\n\\nExtracted Financials:\\n{financial_fact_to_markdown(announcement_fin)}"
            
        report_md = st.session_state.stock_research_report_summary or ""
        report_event = st.session_state.get("stock_research_selected_report_event")
        report_fin = st.session_state.get("stock_research_report_financials")
        if report_event:
            report_md += f"\\n\\nStructured Report Signals:\\n{event_to_markdown(report_event)}"
        if report_fin:
            report_md += f"\\n\\nExtracted Financials:\\n{financial_fact_to_markdown(report_fin)}"

        deps_hash = _safe_hash(announcement_md, report_md)
        cache_key = build_stock_cache_key(company_name, final_symbol, deps_hash)
        cached = load_stock_ai_view(cache_key)

        if cached:
            ai_view = cached.get("ai_view", "")
            st.session_state.stock_research_ai_view_cache_status = "Loaded from AI cache"
        else:
            with st.spinner("Generating stock AI view..."):
                ai_view = generate_stock_ai_view(
                    company_name=company_name,
                    ticker=final_symbol,
                    quote_snapshot=quote_snapshot,
                    market_stats=market_stats,
                    latest_announcement_summary=announcement_md,
                    latest_report_summary=report_md,
                )
            save_stock_ai_view(
                cache_key=cache_key,
                ai_view=ai_view,
                meta={"company_name": company_name, "ticker": final_symbol}
            )
            st.session_state.stock_research_ai_view_cache_status = "Fresh analysis generated"

        st.session_state.stock_research_ai_view = ai_view

    if st.session_state.stock_research_ai_view:
        cache_status = st.session_state.get("stock_research_ai_view_cache_status", "Fresh analysis generated")
        st.caption(f"Status: **{cache_status}**")
        st.write(st.session_state.stock_research_ai_view)
    else:
        st.info("Generate the AI View after analyzing at least one announcement or report for richer output.")