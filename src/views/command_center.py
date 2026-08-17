from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.yahoo_prices import YahooCSEClient
from src.cse_announcements import CSEAnnouncementsClient
from src.alerts_engine import evaluate_alerts
from src.ta_engine import generate_technical_signals
from src.portfolio_risk import evaluate_portfolio_risk_metrics
from src.app_state import (
    get_active_company_name,
    get_active_symbol,
    send_to_analyst_workspace,
    send_to_announcements,
    send_to_stock_research,
)
from src.ui import (
    context_bar,
    divider_label,
    empty_state,
    info_card,
    inject_global_styles,
    page_header,
    render_company_selector,
    section_header,
    status_badge,
)




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"
ALERTS_FILE = BASE_DIR / "data" / "alerts_store.json"

client = YahooCSEClient(universe_path=UNIVERSE_PATH)


@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    try:
        return YahooCSEClient(universe_path=path).load_universe()
    except Exception:
        return pd.DataFrame(columns=["symbol", "company_name"])


@st.cache_data(ttl=300)
def load_announcements_cached() -> pd.DataFrame:
    try:
        return CSEAnnouncementsClient(timeout=4).fetch_announcements("All")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def evaluate_alerts_cached(path: str, file_path: str):
    try:
        return evaluate_alerts(universe_path=path, file_path=file_path)
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=300)
def get_history_cached(symbol: str, period: str = "6m") -> pd.DataFrame:
    try:
        return YahooCSEClient(universe_path=UNIVERSE_PATH).get_history(symbol, period=period)
    except Exception:
        return pd.DataFrame()


def score_importance(title: str, category: str) -> str:
    text = f"{title} {category}".upper()
    high_keywords = [
        "RIGHTS ISSUE", "ACQUISITION", "MERGER", "TAKEOVER", "DIVIDEND",
        "INTERIM FINANCIAL STATEMENTS", "ANNUAL REPORT", "PROFIT WARNING",
        "BOARD MEETING", "SHARE SPLIT", "DELIST", "BONUS ISSUE", "MATERIAL", "CAPITAL",
    ]
    if any(k in text for k in high_keywords):
        return "High"
    return "Other"


def map_company_to_ticker(company_name: str, universe_df: pd.DataFrame) -> str:
    if not company_name or universe_df.empty:
        return ""
    target = str(company_name).strip().upper()
    exact = universe_df[universe_df["company_name"].str.upper() == target]
    if not exact.empty:
        return exact.iloc[0]["symbol"]
    root = (
        target.replace(" PLC", "").replace(" LIMITED", "")
        .replace(" LTD", "").replace(" THE ", " ").strip()
    )
    broad = universe_df[universe_df["company_name"].str.upper().str.contains(root, na=False)]
    if not broad.empty:
        return broad.iloc[0]["symbol"]
    return ""


def _fmt_num(value, decimals=2):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_pct(value):
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):,.2f}%"
    except Exception:
        return "N/A"


# ─── Data Loading ───────────────────────────────────────
universe_df = load_universe_cached(str(UNIVERSE_PATH))
ann_df = load_announcements_cached()

if isinstance(ann_df, pd.DataFrame) and not ann_df.empty:
    ann_df = ann_df.copy()
    ann_df["mapped_ticker"] = ann_df["company_name"].apply(
        lambda x: map_company_to_ticker(str(x), universe_df)
    )
    ann_df["importance_label"] = ann_df.apply(
        lambda row: score_importance(
            str(row.get("announcement_title", "")),
            str(row.get("category", "")),
        ),
        axis=1,
    )
    if "announcement_date_parsed" in ann_df.columns:
        ann_df = ann_df.sort_values(
            by="announcement_date_parsed", ascending=False, na_position="last"
        ).reset_index(drop=True)
else:
    ann_df = pd.DataFrame()

alerts_df, triggered_df = evaluate_alerts_cached(
    path=str(UNIVERSE_PATH),
    file_path=str(ALERTS_FILE),
)

portfolio_snapshot_df = st.session_state.get("portfolio_snapshot_df", pd.DataFrame())
portfolio_market_value = None
portfolio_top_weight = None
if isinstance(portfolio_snapshot_df, pd.DataFrame) and not portfolio_snapshot_df.empty:
    portfolio_market_value = portfolio_snapshot_df["market_value"].fillna(0).sum()
    if "weight_pct" in portfolio_snapshot_df.columns and not portfolio_snapshot_df["weight_pct"].dropna().empty:
        portfolio_top_weight = float(portfolio_snapshot_df["weight_pct"].dropna().max())

# ─── Hero Header ────────────────────────────────────────
page_header(
    "CSE Command Center",
    "Executive market dashboard — select any company for live technical indicators, recent disclosures, and monitoring.",
)

active_symbol = get_active_symbol()
active_company = get_active_company_name()
context_bar(active_symbol, active_company)

# ─── Top Executive KPI Cards ────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Listed Companies", len(universe_df) if isinstance(universe_df, pd.DataFrame) else 294)
kpi2.metric("Latest Disclosures", len(ann_df) if isinstance(ann_df, pd.DataFrame) else 0)
kpi3.metric("Triggered Alerts", len(triggered_df), f"{len(alerts_df)} Active Rules")
kpi4.metric(
    "Portfolio Value",
    f"LKR {_fmt_num(portfolio_market_value)}" if portfolio_market_value else "—",
)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Content Layout (65% Left / 35% Right) ────────
main_left, main_right = st.columns([1.8, 1.0])

with main_left:
    section_header("Equity Research Launcher")

    final_symbol, company_name = render_company_selector(universe_df, key_prefix="cc_launcher")

    if final_symbol:
        qcol1, qcol2, qcol3 = st.columns(3)
        if qcol1.button("Stock Research Hub", use_container_width=True, key="cc_stock_btn", type="primary"):
            send_to_stock_research(final_symbol, company_name or final_symbol)
        if qcol2.button("Company Disclosures", use_container_width=True, key="cc_ann_btn"):
            send_to_announcements(company_name, final_symbol)
        if qcol3.button("Ask AI Copilot", use_container_width=True, key="cc_copilot_btn"):
            send_to_analyst_workspace(
                company_name or final_symbol, final_symbol,
                analysis_mode="News Summary",
                query=f"Build a CSE research snapshot for {company_name or final_symbol}.",
            )

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("High-Priority Disclosures Feed")

    if ann_df.empty:
        empty_state("", "No disclosures loaded", "Disclosures will appear here when fetched from the CSE.")
    else:
        high_df = ann_df[ann_df["importance_label"] == "High"].copy()
        display_df = high_df if not high_df.empty else ann_df
        for card_idx, (_, row) in enumerate(display_df.head(6).iterrows()):
            co = str(row.get("company_name", "")).strip()
            tk = str(row.get("mapped_ticker", "")).strip()
            ttl = str(row.get("announcement_title", "")).strip()
            dt = str(row.get("announcement_date", "")).strip()
            cat = str(row.get("category", "")).strip()
            detail_url = str(row.get("detail_url", "")).strip()
            pdf_url = str(row.get("pdf_url", "")).strip()

            with st.container(border=True):
                c_top1, c_top2 = st.columns([3, 1])
                c_top1.markdown(f"**{co or 'CSE Listed Company'}** &nbsp;<span class='cc-subtle'>({tk or 'CSE'})</span>", unsafe_allow_html=True)
                c_top2.markdown(status_badge("High Priority", "high"), unsafe_allow_html=True)
                st.caption(f"{dt} · {cat}")
                st.write(ttl)
                if detail_url or pdf_url:
                    st.link_button("Open Official Announcement PDF", detail_url or pdf_url)

with main_right:
    section_header("Quick Navigation")

    pages_map = [
        ("Stock Research", "Analyze stocks, financials, and technical charts", "src/views/stock_research.py"),
        ("Document Intelligence", "Extract tables, facts, and interim key figures", "src/views/report_intelligence.py"),
        ("Portfolio & Screener", "Track holdings, risk metrics, and stock screeners", "src/views/portfolio_intelligence.py"),
        ("Copilot & Monitoring", "AI agent memo workspace and system alerts", "src/views/analyst_workspace.py"),
    ]
    for i, (label, desc, target) in enumerate(pages_map):
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(desc)
            if st.button(f"Go to {label}", key=f"nav_btn_{i}", use_container_width=True):
                st.switch_page(target)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Triggered Alerts")

    if triggered_df.empty:
        empty_state("", "All Clear", "No monitoring rule thresholds triggered.")
    else:
        for a_idx, (_, row) in enumerate(triggered_df.head(4).iterrows()):
            co = str(row.get("company_name", "")).strip()
            tk = str(row.get("canonical_symbol", "")).strip()
            reason = str(row.get("reason", "")).strip()

            with st.container(border=True):
                st.markdown(f"**{co or tk}** {status_badge('Alert', 'triggered')}", unsafe_allow_html=True)
                st.caption(reason)
                if st.button("Investigate in Copilot", key=f"cc_al_cp_{a_idx}", use_container_width=True):
                    send_to_analyst_workspace(
                        co or tk, tk,
                        analysis_mode="Catalysts & Risks",
                        query=f"Review this alert for {co or tk}: {reason}",
                    )

# ─── Bottom Row ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
divider_label("Portfolio & Alerts Overview")
bot_left, bot_right = st.columns(2)

with bot_left:
    section_header("Portfolio Snapshot")
    if isinstance(portfolio_snapshot_df, pd.DataFrame) and not portfolio_snapshot_df.empty:
        preview_cols = [
            c for c in ["canonical_symbol", "company_name", "market_value", "weight_pct", "unrealized_pnl_pct"]
            if c in portfolio_snapshot_df.columns
        ]
        st.dataframe(
            portfolio_snapshot_df[preview_cols].head(8),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Open Portfolio Intelligence", use_container_width=True, key="cc_port_btn"):
            st.switch_page("src/views/portfolio_intelligence.py")
    else:
        empty_state("", "No portfolio loaded", "Upload your holdings CSV in Portfolio Intelligence to see your snapshot here.")
        if st.button("Build Portfolio Snapshot", use_container_width=True, key="cc_port_build"):
            st.switch_page("src/views/portfolio_intelligence.py")

with bot_right:
    section_header("Alert Rules")
    if alerts_df.empty:
        empty_state("", "No alerts configured", "Set up price and disclosure alerts in the Alerts & Monitoring page.")
        if st.button("Configure Alerts", use_container_width=True, key="cc_alerts_btn"):
            st.switch_page("src/views/alerts_monitoring.py")
    else:
        preview_cols = [
            c for c in ["company_name", "canonical_symbol", "rule_type", "is_enabled", "last_triggered_at"]
            if c in alerts_df.columns
        ]
        st.dataframe(
            alerts_df[preview_cols].head(8),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Manage Alerts & Monitoring", use_container_width=True, key="cc_alerts_manage"):
            st.switch_page("src/views/alerts_monitoring.py")
