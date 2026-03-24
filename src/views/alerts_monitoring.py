from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from src.yahoo_prices import YahooCSEClient
from src.alerts_engine import (
    DEFAULT_ALERTS_FILE,
    RULE_TYPES,
    add_alert,
    delete_alert,
    evaluate_alerts,
    load_alerts,
    update_alert,
)
from src.ui import inject_global_styles, page_header
from src.app_state import send_to_analyst_workspace, send_to_stock_research




inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"
ALERTS_FILE = BASE_DIR / "data" / "alerts_store.json"

client = YahooCSEClient(universe_path=UNIVERSE_PATH)

page_header(
    "Alerts & Monitoring",
    "Create alert rules for price, momentum, liquidity, and recent CSE disclosures.",
)
st.info("This first version evaluates alerts on page load and refresh. Background notifications can come later.")

@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    return YahooCSEClient(universe_path=path).load_universe()


def threshold_label(rule_key: str) -> str:
    if rule_key in {"PRICE_ABOVE", "PRICE_BELOW"}:
        return "Threshold Price"
    if rule_key == "RETURN_1M_ABOVE":
        return "Threshold Return %"
    if rule_key == "AVG_VOLUME_20D_ABOVE":
        return "Threshold Avg 20D Volume"
    return "Threshold"


universe_df = load_universe_cached(str(UNIVERSE_PATH))

# ---------------------------
# Build alert
# ---------------------------
st.subheader("Create Alert")

builder_col1, builder_col2 = st.columns([2, 1])

search_text = builder_col1.text_input(
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

selected_label = builder_col1.selectbox(
    "Pick company",
    options=[""] + list(option_map.keys()),
    index=0,
)

manual_symbol = builder_col2.text_input(
    "Or type symbol / alias",
    placeholder="e.g. JKH, JKH.N, JKH.N0000",
)

typed_symbol = client.normalize_symbol_text(manual_symbol)
selected_symbol = option_map.get(selected_label, "")
canonical_symbol = (
    client.resolve_symbol_from_universe(typed_symbol, universe_df)
    if typed_symbol
    else selected_symbol
)

company_name = client.get_company_name(canonical_symbol, universe_df) if canonical_symbol else ""

rule_col1, rule_col2, rule_col3 = st.columns([2, 1, 1])

selected_rule_key = rule_col1.selectbox(
    "Rule Type",
    options=list(RULE_TYPES.keys()),
    format_func=lambda k: RULE_TYPES[k],
)

needs_threshold = selected_rule_key in {
    "PRICE_ABOVE",
    "PRICE_BELOW",
    "RETURN_1M_ABOVE",
    "AVG_VOLUME_20D_ABOVE",
}

threshold_value = None
if needs_threshold:
    default_value = 0.0 if selected_rule_key != "AVG_VOLUME_20D_ABOVE" else 100000.0
    threshold_value = rule_col2.number_input(
        threshold_label(selected_rule_key),
        min_value=0.0,
        value=float(default_value),
        step=1.0,
    )
else:
    rule_col2.write("")
    rule_col3.write("")

notes = st.text_input(
    "Notes (optional)",
    placeholder="e.g. Monitor breakout above resistance / watch for new disclosures",
)

if st.button("Add Alert", use_container_width=True):
    if not canonical_symbol:
        st.warning("Select a company or type a valid symbol first.")
    else:
        add_alert(
            symbol=typed_symbol or canonical_symbol,
            company_name=company_name or canonical_symbol,
            canonical_symbol=canonical_symbol,
            rule_type=selected_rule_key,
            threshold_value=threshold_value if needs_threshold else None,
            notes=notes,
            file_path=ALERTS_FILE,
        )
        st.success("Alert added.")
        st.rerun()

# ---------------------------
# Evaluate alerts
# ---------------------------
refresh_col1, refresh_col2 = st.columns([1, 3])
if refresh_col1.button("Refresh Alerts", use_container_width=True):
    st.rerun()

with st.spinner("Evaluating alerts..."):
    alerts_df, triggered_df = evaluate_alerts(
        universe_path=UNIVERSE_PATH,
        file_path=ALERTS_FILE,
    )

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
metric_col1.metric("Active Alerts", int(alerts_df["is_enabled"].sum()) if not alerts_df.empty else 0)
metric_col2.metric("Triggered Now", len(triggered_df))
metric_col3.metric(
    "Price Rules",
    int(alerts_df["rule_type"].isin(["PRICE_ABOVE", "PRICE_BELOW"]).sum()) if not alerts_df.empty else 0,
)
metric_col4.metric(
    "Disclosure Rules",
    int(alerts_df["rule_type"].isin(["ANY_DISCLOSURE_7D", "HIGH_PRIORITY_DISCLOSURE_30D"]).sum()) if not alerts_df.empty else 0,
)

tabs = st.tabs(["Triggered Alerts", "Configured Alerts", "Manage Alerts", "Debug"])

with tabs[0]:
    st.subheader("Triggered Alerts")

    if triggered_df.empty:
        st.info("No alerts are triggered right now.")
    else:
        for idx, row in triggered_df.iterrows():
            company_name = str(row.get("company_name", "")).strip()
            symbol = str(row.get("canonical_symbol", "")).strip()
            rule_type = str(row.get("rule_type", "")).strip()
            reason = str(row.get("reason", "")).strip()
            current_value = row.get("current_value", None)

            with st.container(border=True):
                st.markdown(f"**{company_name or symbol}**")
                st.caption(f"{symbol} • {RULE_TYPES.get(rule_type, rule_type)}")
                st.write(reason)

                if pd.notna(current_value):
                    st.caption(f"Current Value: {current_value}")

                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button(
                    "Open Stock Research",
                    key=f"open_triggered_{idx}",
                    use_container_width=True,
                ):
                    send_to_stock_research(symbol)

                if btn_col2.button(
                    "Send to Analyst Workspace",
                    key=f"analyst_triggered_{idx}",
                    use_container_width=True,
                ):
                    send_to_analyst_workspace(
                        company_name=company_name,
                        ticker=symbol,
                        analysis_mode="Catalysts & Risks",
                        query=f"Review this CSE alert for {company_name or symbol} and explain the investment implications, risks, and what to monitor next.",
                    )

with tabs[1]:
    st.subheader("Configured Alerts")

    if alerts_df.empty:
        st.info("No alerts configured yet.")
    else:
        display_df = alerts_df.copy()
        display_df["rule_label"] = display_df["rule_type"].map(RULE_TYPES).fillna(display_df["rule_type"])
        st.dataframe(
            display_df[
                [
                    "id",
                    "company_name",
                    "canonical_symbol",
                    "rule_label",
                    "threshold_value",
                    "is_enabled",
                    "created_at",
                    "last_triggered_at",
                    "notes",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tabs[2]:
    st.subheader("Manage Alerts")

    alerts_list = load_alerts(ALERTS_FILE)
    if not alerts_list:
        st.info("No alerts to manage.")
    else:
        manage_map = {
            f"{a.get('company_name', a.get('canonical_symbol', ''))} | {RULE_TYPES.get(a.get('rule_type', ''), a.get('rule_type', ''))} | {a.get('id')}": a
            for a in alerts_list
        }

        selected_manage_label = st.selectbox(
            "Select alert",
            options=[""] + list(manage_map.keys()),
            index=0,
        )

        if selected_manage_label:
            selected_alert = manage_map[selected_manage_label]
            st.write(selected_alert)

            mcol1, mcol2, mcol3 = st.columns(3)

            if selected_alert.get("is_enabled"):
                if mcol1.button("Disable Alert", use_container_width=True):
                    update_alert(
                        selected_alert["id"],
                        {"is_enabled": False},
                        file_path=ALERTS_FILE,
                    )
                    st.rerun()
            else:
                if mcol1.button("Enable Alert", use_container_width=True):
                    update_alert(
                        selected_alert["id"],
                        {"is_enabled": True},
                        file_path=ALERTS_FILE,
                    )
                    st.rerun()

            if mcol2.button("Delete Alert", use_container_width=True):
                delete_alert(selected_alert["id"], file_path=ALERTS_FILE)
                st.rerun()

            if mcol3.button("Open Stock Research", use_container_width=True):
                send_to_stock_research(selected_alert["canonical_symbol"])

with tabs[3]:
    st.subheader("Debug")
    st.write(f"Universe rows: {len(universe_df)}")
    st.write(f"Alerts rows: {len(alerts_df)}")
    st.write(f"Triggered rows: {len(triggered_df)}")

    if not alerts_df.empty:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

    if not triggered_df.empty:
        st.dataframe(triggered_df, use_container_width=True, hide_index=True)