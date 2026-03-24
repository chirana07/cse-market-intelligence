from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.yahoo_prices import YahooCSEClient
from src.cse_announcements import CSEAnnouncementsClient
from src.alerts_engine import evaluate_alerts
from src.app_state import (
    get_active_company_name,
    get_active_symbol,
    send_to_analyst_workspace,
    send_to_announcements,
    send_to_stock_research,
)
from src.ui import (
    inject_global_styles,
    page_header,
    section_header,
    info_card,
    status_badge,
    empty_state,
    context_bar,
    divider_label,
)




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"
ALERTS_FILE = BASE_DIR / "data" / "alerts_store.json"

client = YahooCSEClient(universe_path=UNIVERSE_PATH)


@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    return YahooCSEClient(universe_path=path).load_universe()


@st.cache_data(ttl=300)
def load_announcements_cached() -> pd.DataFrame:
    return CSEAnnouncementsClient().fetch_announcements("All")


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

if not ann_df.empty:
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

alerts_df, triggered_df = evaluate_alerts(
    universe_path=UNIVERSE_PATH,
    file_path=ALERTS_FILE,
)

portfolio_snapshot_df = st.session_state.get("portfolio_snapshot_df", pd.DataFrame())
portfolio_market_value = None
portfolio_top_weight = None
if isinstance(portfolio_snapshot_df, pd.DataFrame) and not portfolio_snapshot_df.empty:
    portfolio_market_value = portfolio_snapshot_df["market_value"].fillna(0).sum()
    if "weight_pct" in portfolio_snapshot_df.columns and not portfolio_snapshot_df["weight_pct"].dropna().empty:
        portfolio_top_weight = float(portfolio_snapshot_df["weight_pct"].dropna().max())

active_symbol = get_active_symbol()
active_company = get_active_company_name()

# ─── Hero Header ────────────────────────────────────────
page_header(
    "CSE Command Center",
    "Your daily starting point — market context, key disclosures, alerts, and portfolio intelligence.",
)

# Quick CTA row
hero_c1, hero_c2, hero_c3, hero_c4 = st.columns([2, 1, 1, 1])
with hero_c2:
    if st.button("Stock Research", use_container_width=True):
        st.switch_page("src/views/stock_research.py")
with hero_c3:
    if st.button("Disclosures", use_container_width=True):
        st.switch_page("src/views/announcements_hub.py")
with hero_c4:
    if st.button("Ask Copilot", use_container_width=True):
        st.switch_page("src/views/analyst_workspace.py")

# ─── Active Context Bar ──────────────────────────────────
context_bar(active_symbol, active_company)

# ─── KPI Strip ──────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Configured Alerts", len(alerts_df))
kpi2.metric("Triggered Now", len(triggered_df))
kpi3.metric("Recent Disclosures", len(ann_df) if isinstance(ann_df, pd.DataFrame) else 0)
kpi4.metric(
    "Portfolio Value",
    f"LKR {_fmt_num(portfolio_market_value)}" if portfolio_market_value else "—",
)
kpi5.metric(
    "Top Holding",
    _fmt_pct(portfolio_top_weight) if portfolio_top_weight else "—",
)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Main Content: Left + Right ──────────────────────────
main_left, main_right = st.columns([3, 2])

with main_left:
    # ── Stock Opener ─────────────────────────────────────
    section_header("Quick Research Launcher")

    search_col1, search_col2 = st.columns([3, 1])

    search_text = search_col1.text_input(
        "Search company or symbol",
        placeholder="e.g. John Keells, JKH, COMB",
        label_visibility="collapsed",
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
        "Pick company",
        options=[""] + list(option_map.keys()),
        index=0,
        label_visibility="collapsed",
        placeholder="Select a company…",
    )

    manual_symbol = search_col2.text_input(
        "Or type symbol",
        placeholder="JKH",
        label_visibility="collapsed",
    )

    typed_symbol = client.normalize_symbol_text(manual_symbol)
    selected_symbol = option_map.get(selected_label, "")
    final_symbol = (
        client.resolve_symbol_from_universe(typed_symbol, universe_df)
        if typed_symbol
        else selected_symbol
    )
    company_name = client.get_company_name(final_symbol, universe_df) if final_symbol else ""

    qcol1, qcol2, qcol3 = st.columns(3)
    if qcol1.button("Stock Research", use_container_width=True, disabled=not final_symbol, key="cc_stock_btn"):
        send_to_stock_research(final_symbol, company_name or final_symbol)
    if qcol2.button("Announcements", use_container_width=True, disabled=not company_name, key="cc_ann_btn"):
        send_to_announcements(company_name, final_symbol)
    if qcol3.button("Ask Copilot", use_container_width=True, disabled=not final_symbol, key="cc_copilot_btn"):
        send_to_analyst_workspace(
            company_name or final_symbol, final_symbol,
            analysis_mode="News Summary",
            query=f"Build a CSE research snapshot for {company_name or final_symbol}.",
        )

    divider_label("High-Priority Disclosures")

    if ann_df.empty:
        empty_state("", "No disclosures loaded", "Disclosures will appear here when fetched from the CSE.")
    else:
        high_df = ann_df[ann_df["importance_label"] == "High"].copy()
        if high_df.empty:
            empty_state("", "No high-priority disclosures", "All clear — no material events detected in the latest feed.")
        else:
            for card_idx, (_, row) in enumerate(high_df.head(7).iterrows()):
                co = str(row.get("company_name", "")).strip()
                tk = str(row.get("mapped_ticker", "")).strip()
                ttl = str(row.get("announcement_title", "")).strip()
                dt = str(row.get("announcement_date", "")).strip()
                cat = str(row.get("category", "")).strip()
                detail_url = str(row.get("detail_url", "")).strip()
                pdf_url = str(row.get("pdf_url", "")).strip()

                with st.container(border=True):
                    badge_html = status_badge("High Priority", "high")
                    st.markdown(
                        f"**{co or 'Unknown Company'}** &nbsp;{badge_html}",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{dt} · {cat} · {tk or 'No ticker'}")
                    st.write(ttl)

                    act_c1, act_c2, act_c3 = st.columns(3)
                    if detail_url:
                        act_c1.link_button("Open ↗", detail_url, use_container_width=True)
                    elif pdf_url:
                        act_c1.link_button("PDF ↗", pdf_url, use_container_width=True)
                    if act_c2.button("Stock Research", key=f"cc_sr_{card_idx}", use_container_width=True, disabled=not tk):
                        send_to_stock_research(tk, co)
                    if act_c3.button("Ask Copilot", key=f"cc_cp_{card_idx}", use_container_width=True, disabled=not tk):
                        send_to_analyst_workspace(
                            co or tk, tk,
                            analysis_mode="Catalysts & Risks",
                            query=f"Analyze this CSE disclosure for {co or tk}: {ttl}",
                        )

with main_right:
    # ── Module Launcher ────────────────────────────────
    section_header("Navigate")

    nav_c1, nav_c2 = st.columns(2)
    pages_map = [
        ("Stock Research", "src/views/stock_research.py"),
        ("Announcements", "src/views/announcements_hub.py"),
        ("Screener", "src/views/stock_screener.py"),
        ("Portfolio", "src/views/portfolio_intelligence.py"),
        ("Reports", "src/views/report_intelligence.py"),
        ("Alerts", "src/views/alerts_monitoring.py"),
        ("Market Data", "src/views/market_dashboard.py"),
        ("Copilot", "src/views/analyst_workspace.py"),
    ]
    for i, (label, target) in enumerate(pages_map):
        col = nav_c1 if i % 2 == 0 else nav_c2
        if col.button(f"{label}", use_container_width=True, key=f"nav_{i}"):
            st.switch_page(target)

    divider_label("Triggered Alerts")

    if triggered_df.empty:
        empty_state("", "No alerts triggered", "All monitoring conditions are within bounds.")
    else:
        for a_idx, (_, row) in enumerate(triggered_df.head(6).iterrows()):
            co = str(row.get("company_name", "")).strip()
            tk = str(row.get("canonical_symbol", "")).strip()
            reason = str(row.get("reason", "")).strip()

            with st.container(border=True):
                badge_html = status_badge("Alert", "triggered")
                st.markdown(
                    f"**{co or tk}** &nbsp;{badge_html}",
                    unsafe_allow_html=True,
                )
                st.caption(tk)
                st.caption(reason)
                al_c1, al_c2 = st.columns(2)
                if al_c1.button("Research", key=f"cc_al_sr_{a_idx}", use_container_width=True):
                    send_to_stock_research(tk, co)
                if al_c2.button("Review", key=f"cc_al_cp_{a_idx}", use_container_width=True):
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
        if st.button("Open Portfolio Intelligence →", use_container_width=True, key="cc_port_btn"):
            st.switch_page("src/views/portfolio_intelligence.py")
    else:
        empty_state("", "No portfolio loaded", "Upload your holdings CSV in Portfolio Intelligence to see your snapshot here.")
        if st.button("Build Portfolio Snapshot →", use_container_width=True, key="cc_port_build"):
            st.switch_page("src/views/portfolio_intelligence.py")

with bot_right:
    section_header("Alert Rules")
    if alerts_df.empty:
        empty_state("", "No alerts configured", "Set up price and disclosure alerts in the Alerts & Monitoring page.")
        if st.button("Configure Alerts →", use_container_width=True, key="cc_alerts_btn"):
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
        if st.button("Manage Alerts & Monitoring →", use_container_width=True, key="cc_alerts_manage"):
            st.switch_page("src/views/alerts_monitoring.py")

with st.expander("🔧 Debug Info"):
    st.caption(f"Universe rows: {len(universe_df)} · Announcements: {len(ann_df)} · Alerts: {len(alerts_df)} · Triggered: {len(triggered_df)}")