from __future__ import annotations

from pathlib import Path
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.portfolio_intelligence import (
    normalize_holdings_csv,
    build_portfolio_snapshot,
    portfolio_summary_metrics,
    concentration_flags,
    generate_portfolio_review,
)
from src.persistence import (
    build_portfolio_cache_key,
    load_portfolio_review,
    save_portfolio_review,
    build_report_cache_key,
    load_report_artifacts,
)
from src.ui import inject_global_styles, page_header, section_header, status_badge, empty_state, divider_label
from src.app_state import send_to_analyst_workspace, send_to_stock_research




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"

if "portfolio_snapshot_df" not in st.session_state:
    st.session_state.portfolio_snapshot_df = pd.DataFrame()

if "portfolio_review_text" not in st.session_state:
    st.session_state.portfolio_review_text = ""


def _fmt_num(value, decimals=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def _pie_chart(df: pd.DataFrame):
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["canonical_symbol"],
                values=df["market_value"].fillna(0),
                hole=0.4,
            )
        ]
    )
    fig.update_layout(
        title="Portfolio Allocation by Holding",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def _bar_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["canonical_symbol"],
            y=df["unrealized_pnl_pct"],
            name="Unrealized PnL %",
        )
    )
    fig.update_layout(
        title="Unrealized PnL % by Holding",
        xaxis_title="Holding",
        yaxis_title="PnL %",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def send_portfolio_to_analyst(snapshot_df: pd.DataFrame):
    symbols = ", ".join(snapshot_df["canonical_symbol"].head(8).tolist())
    send_to_analyst_workspace(
        company_name="Portfolio Review",
        ticker="",
        analysis_mode="Portfolio Memo",
        query=f"Review this CSE portfolio and provide concentration risks, strengths, weaknesses, and monitoring priorities. Holdings include: {symbols}",
    )


page_header(
    "Portfolio Intelligence",
    "Upload your CSE holdings, analyze allocation and concentration risk, and get an AI portfolio review.",
)

upload_col1, upload_col2 = st.columns([2, 1])

uploaded_csv = upload_col1.file_uploader(
    "Upload holdings CSV",
    type=["csv"],
)

with upload_col2:
    st.markdown("**Expected CSV columns**")
    st.code("symbol,quantity,avg_cost", language="text")
    st.caption("Example: JKH.N0000,100,175.50")

manual_csv_text = st.text_area(
    "Or paste holdings CSV content",
    placeholder="symbol,quantity,avg_cost\nJKH.N0000,100,175.50\nCOMB.N0000,80,115.00",
    height=140,
)

if st.button("Build Portfolio Snapshot", use_container_width=True):
    try:
        if uploaded_csv is not None:
            raw_df = pd.read_csv(uploaded_csv)
        elif manual_csv_text.strip():
            raw_df = pd.read_csv(StringIO(manual_csv_text))
        else:
            raw_df = pd.DataFrame()

        holdings_df = normalize_holdings_csv(raw_df)

        if holdings_df.empty:
            st.warning("No valid holdings were found. Make sure columns include symbol, quantity, avg_cost.")
        else:
            with st.spinner("Building portfolio snapshot..."):
                snapshot_df = build_portfolio_snapshot(
                    holdings_df=holdings_df,
                    universe_path=UNIVERSE_PATH,
                )

            st.session_state.portfolio_snapshot_df = snapshot_df
            st.session_state.portfolio_review_text = ""
    except Exception as e:
        st.error(f"Failed to build portfolio snapshot: {e}")

snapshot_df = st.session_state.portfolio_snapshot_df

if snapshot_df.empty:
    empty_state(
        "",
        "No portfolio loaded yet",
        "Upload your holdings CSV above or paste the data directly to build your portfolio snapshot.",
    )
    st.stop()

metrics = portfolio_summary_metrics(snapshot_df)
flags = concentration_flags(snapshot_df)

metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Holdings", metrics["holdings"])
metric_col2.metric("Market Value", _fmt_num(metrics["total_market_value"]))
metric_col3.metric("Cost Basis", _fmt_num(metrics["total_cost_basis"]))
metric_col4.metric("Unrealized PnL", _fmt_num(metrics["total_unrealized_pnl"]))
metric_col5.metric("Unrealized PnL %", _fmt_num(metrics["total_unrealized_pnl_pct"]))

flag_col1, flag_col2 = st.columns([2, 1])

with flag_col1:
    st.subheader("Concentration & Risk Flags")
    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No major concentration flags detected from the current rule set.")

with flag_col2:
    if st.button("Send Portfolio to Analyst Workspace", use_container_width=True):
        send_portfolio_to_analyst(snapshot_df)

tabs = st.tabs(["Overview", "Holdings Table", "AI Review", "Debug"])

with tabs[0]:
    top_col1, top_col2 = st.columns(2)

    with top_col1:
        st.plotly_chart(
            _pie_chart(snapshot_df),
            use_container_width=True,
            key="portfolio_allocation_chart",
        )

    with top_col2:
        st.plotly_chart(
            _bar_chart(snapshot_df),
            use_container_width=True,
            key="portfolio_pnl_chart",
        )

    st.subheader("Top Holdings")
    top_holdings = snapshot_df.sort_values("market_value", ascending=False).head(10).copy()
    st.dataframe(
        top_holdings[
            [
                "canonical_symbol",
                "company_name",
                "weight_pct",
                "market_value",
                "unrealized_pnl_pct",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Holdings Table")

    st.dataframe(
        snapshot_df[
            [
                "symbol",
                "canonical_symbol",
                "company_name",
                "quantity",
                "avg_cost",
                "last_price",
                "market_value",
                "cost_basis",
                "unrealized_pnl",
                "unrealized_pnl_pct",
                "weight_pct",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    option_map = {
        f"{row['company_name']} ({row['canonical_symbol']})": row["canonical_symbol"]
        for _, row in snapshot_df.iterrows()
    }

    selected_label = st.selectbox(
        "Open a holding in Stock Research",
        options=[""] + list(option_map.keys()),
        index=0,
    )

    if selected_label:
        selected_symbol = option_map[selected_label]
        if st.button("Open Selected Holding", use_container_width=True):
            send_to_stock_research(selected_symbol)

        # --- Light financial signals from cache (no LLM call) ---
        selected_row = snapshot_df[snapshot_df["canonical_symbol"] == selected_symbol]
        company_name_sel = selected_row.iloc[0].get("company_name", "") if not selected_row.empty else ""
        cached_report = load_report_artifacts(
            build_report_cache_key(company_name_sel, selected_symbol, selected_symbol)
        )
        cached_financials = cached_report.get("financials") if cached_report else None
        if cached_financials:
            with st.expander(f"Cached Financial Signals: {selected_symbol}"):
                hf1, hf2, hf3 = st.columns(3)
                hf1.write(f"**Tone**: {cached_financials.get('management_tone', 'N/A')}")
                hf2.write(f"**Period**: {cached_financials.get('reporting_period', 'N/A')}")
                hf3.write(f"**EPS**: {cached_financials.get('eps', 'N/A')}")
                pos = cached_financials.get("positive_signals", [])
                if pos:
                    st.write("**Positives**: " + "; ".join(str(p) for p in pos[:3]))
                risk = cached_financials.get("risk_signals", [])
                if risk:
                    st.write("**Risks**: " + "; ".join(str(r) for r in risk[:3]))
                payout = cached_financials.get("payout_signal")
                if payout and payout != "Unknown":
                    st.write(f"**Dividend**: {payout}")
        elif not cached_financials:
            st.caption(f"No cached financial signals for {selected_symbol}. Analyze a report in Report Intelligence first.")

with tabs[2]:
    st.subheader("AI Portfolio Review")

    if st.button("Generate Portfolio Review", use_container_width=True):
        cache_key = build_portfolio_cache_key(snapshot_df)
        cached = load_portfolio_review(cache_key)
        
        if cached:
            review_text = cached.get("review", "")
            st.session_state.portfolio_review_cache_status = "Loaded from AI cache"
        else:
            with st.spinner("Generating AI portfolio review..."):
                review_text = generate_portfolio_review(snapshot_df)
            save_portfolio_review(cache_key, review_text, meta={"holdings_count": len(snapshot_df)})
            st.session_state.portfolio_review_cache_status = "Fresh analysis generated"

        st.session_state.portfolio_review_text = review_text

    if st.session_state.portfolio_review_text:
        cache_status = st.session_state.get("portfolio_review_cache_status", "Fresh analysis generated")
        st.caption(f"Status: **{cache_status}**")
        st.write(st.session_state.portfolio_review_text)
    else:
        st.info("Generate an AI review to get portfolio insights.")

with tabs[3]:
    st.subheader("Debug")
    st.write(f"Rows: {len(snapshot_df)}")
    st.dataframe(snapshot_df.head(10), use_container_width=True, hide_index=True)