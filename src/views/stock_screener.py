from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.yahoo_prices import YahooCSEClient
from src.cse_announcements import CSEAnnouncementsClient
from src.screener_utils import build_screening_dataset, apply_nl_screener_hint
from src.ui import inject_global_styles, page_header, section_header, status_badge, empty_state, divider_label
from src.app_state import send_to_analyst_workspace, send_to_stock_research
from src.persistence import build_report_cache_key, load_report_artifacts



inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"

client = YahooCSEClient(universe_path=UNIVERSE_PATH)
ann_client = CSEAnnouncementsClient()

page_header(
    "CSE Stock Screener",
    "Filter the CSE universe by price action, liquidity, and disclosure signals to surface research ideas.",
)

if "screener_results_df" not in st.session_state:
    st.session_state.screener_results_df = pd.DataFrame()


@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    return YahooCSEClient(universe_path=path).load_universe()


@st.cache_data(ttl=300)
def load_announcements_cached() -> pd.DataFrame:
    return CSEAnnouncementsClient().fetch_announcements("All")


@st.cache_data(ttl=900)
def build_screening_dataset_cached(universe_path: str, row_limit: int) -> pd.DataFrame:
    universe_df = YahooCSEClient(universe_path=universe_path).load_universe()
    announcements_df = CSEAnnouncementsClient().fetch_announcements("All")
    return build_screening_dataset(
        universe_df=universe_df,
        universe_path=universe_path,
        announcements_df=announcements_df,
        limit=row_limit,
    )


def _fmt_num(x, decimals=2):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return "N/A"


universe_df = load_universe_cached(str(UNIVERSE_PATH))
announcements_df = load_announcements_cached()

top_col1, top_col2 = st.columns([2, 1])
top_col1.write(f"Universe size: **{len(universe_df)}** companies")
row_limit = top_col2.selectbox("Universe slice to screen", [10, 20, 30, 50], index=1)

screen_df = build_screening_dataset_cached(str(UNIVERSE_PATH), row_limit)

filter_col1, filter_col2, filter_col3 = st.columns(3)

min_1m_return = filter_col1.slider("Minimum 1M Return %", -50, 100, 0)
min_3m_return = filter_col2.slider("Minimum 3M Return %", -50, 150, 0)
max_volatility = filter_col3.slider("Maximum Volatility %", 10, 150, 80)

filter_col4, filter_col5, filter_col6 = st.columns(3)
min_avg_volume = filter_col4.number_input("Minimum Avg 20D Volume", min_value=0, value=0, step=50000)
require_disclosures = filter_col5.toggle("Require recent disclosures", value=False)
require_high_priority = filter_col6.toggle("Require high-priority disclosures", value=False)

nl_prompt = st.text_input(
    "Natural-language screener prompt",
    placeholder="e.g. find liquid CSE names with positive momentum and recent disclosures",
)

if st.button("Run Screener", use_container_width=True):
    work = screen_df.copy()

    work = work[work["return_1m_pct"].fillna(-999) >= min_1m_return]
    work = work[work["return_3m_pct"].fillna(-999) >= min_3m_return]
    work = work[work["volatility_pct"].fillna(999) <= max_volatility]
    work = work[work["avg_volume_20d"].fillna(0) >= min_avg_volume]

    if require_disclosures:
        work = work[work["announcement_count"] > 0]

    if require_high_priority:
        work = work[work["high_priority_count"] > 0]

    work = apply_nl_screener_hint(work, nl_prompt)

    work = work.sort_values(
        by=["return_3m_pct", "announcement_count", "avg_volume_20d"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    st.session_state.screener_results_df = work

result_df = st.session_state.screener_results_df

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Candidates", len(result_df))
metric_col2.metric("With Disclosures", int((result_df["announcement_count"] > 0).sum()) if not result_df.empty else 0)
metric_col3.metric("High-Priority Disclosure", int((result_df["high_priority_count"] > 0).sum()) if not result_df.empty else 0)
metric_col4.metric("Universe Slice", row_limit)

if result_df.empty:
    st.info("Run the screener to generate ideas.")
else:
    st.subheader("Screening Results")

    display_df = result_df.copy()
    st.dataframe(
        display_df[
            [
                "symbol",
                "company_name",
                "last_price",
                "return_1m_pct",
                "return_3m_pct",
                "return_6m_pct",
                "volatility_pct",
                "avg_volume_20d",
                "announcement_count",
                "high_priority_count",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Action Panel")

    option_map = {
        f"{row['company_name']} ({row['symbol']})": row["symbol"]
        for _, row in result_df.iterrows()
    }

    selected_label = st.selectbox(
        "Pick a candidate",
        options=[""] + list(option_map.keys()),
        index=0,
    )

    if selected_label:
        selected_symbol = option_map[selected_label]
        selected_row = result_df[result_df["symbol"] == selected_symbol].iloc[0]

        detail_col1, detail_col2 = st.columns(2)
        detail_col1.markdown(f"**{selected_row['company_name']}**")
        detail_col1.caption(f"Ticker: {selected_row['symbol']}")
        detail_col1.write(f"Last Price: {_fmt_num(selected_row['last_price'])}")
        detail_col1.write(f"1M Return: {_fmt_num(selected_row['return_1m_pct'])}%")
        detail_col1.write(f"3M Return: {_fmt_num(selected_row['return_3m_pct'])}%")
        detail_col1.write(f"6M Return: {_fmt_num(selected_row['return_6m_pct'])}%")

        detail_col2.write(f"Volatility: {_fmt_num(selected_row['volatility_pct'])}%")
        detail_col2.write(f"Avg 20D Volume: {_fmt_num(selected_row['avg_volume_20d'], 0)}")
        detail_col2.write(f"Announcements: {int(selected_row['announcement_count'])}")
        detail_col2.write(f"High-Priority Announcements: {int(selected_row['high_priority_count'])}")

        # --- Light financial signals from cache (no LLM call) ---
        company_name_sel = selected_row.get("company_name", "")
        # Attempt a broad cache key using just symbol (label-agnostic)
        cached_report = load_report_artifacts(
            build_report_cache_key(company_name_sel, selected_symbol, selected_symbol)
        )
        cached_financials = cached_report.get("financials") if cached_report else None
        if cached_financials:
            with st.expander("Cached Financial Signals (from last analyzed report)"):
                fin_col1, fin_col2, fin_col3 = st.columns(3)
                fin_col1.write(f"**Tone**: {cached_financials.get('management_tone', 'N/A')}")
                fin_col2.write(f"**Period**: {cached_financials.get('reporting_period', 'N/A')}")
                fin_col3.write(f"**Confidence**: {cached_financials.get('confidence', 'N/A')}")
                payout = cached_financials.get("payout_signal")
                if payout and payout != "Unknown":
                    st.write(f"**Dividend / Payout**: {payout}")
                guidance = cached_financials.get("guidance_signal")
                if guidance and guidance != "Unknown":
                    st.write(f"**Guidance**: {guidance}")
                risks = cached_financials.get("risk_signals", [])
                if risks:
                    st.caption("Risk signals: " + "; ".join(str(r) for r in risks[:3]))

        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Open Stock Research", use_container_width=True):
            send_to_stock_research(
                symbol=selected_symbol,
                company_name=selected_row["company_name"],
            )

        if action_col2.button("Send to Analyst Workspace", use_container_width=True):
            send_to_analyst_workspace(
                company_name=selected_row["company_name"],
                ticker=selected_row["symbol"],
                analysis_mode="Bull vs Bear Case",
                query=f"Evaluate whether {selected_row['company_name'] or selected_row['symbol']} looks interesting based on price action, disclosures, and risk/reward.",
            )

with st.expander("Debug Preview"):
    st.write(f"Universe rows: {len(universe_df)}")
    st.write(f"Announcements rows: {len(announcements_df)}")
    st.write(f"Screening dataset rows: {len(screen_df)}")
    if not screen_df.empty:
        st.dataframe(screen_df.head(10), use_container_width=True, hide_index=True)