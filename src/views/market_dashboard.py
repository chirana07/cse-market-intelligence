from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.yahoo_prices import YahooCSEClient
from src.ui import inject_global_styles, page_header
from src.app_state import send_to_analyst_workspace, send_to_stock_research, set_active_symbol



inject_global_styles()

BASE_DIR = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = BASE_DIR / "data" / "cse_universe.csv"
client = YahooCSEClient(universe_path=UNIVERSE_PATH)

page_header(
    "CSE Market Dashboard",
    "Single-stock lookup and watchlists powered by Yahoo Finance / yfinance for Colombo delayed quotes."
)

st.info(
    "This page uses Yahoo Finance symbols like JKH-N0000.CM for Colombo delayed quotes. "
    "Use the button in the Single Stock tab to send a selected company into the Analyst Workspace."
)


@st.cache_data(ttl=3600)
def load_universe_cached(path: str) -> pd.DataFrame:
    return YahooCSEClient(universe_path=path).load_universe()


@st.cache_data(ttl=900)
def load_history_cached(symbol: str, universe_path: str, period: str, interval: str) -> pd.DataFrame:
    local_client = YahooCSEClient(universe_path=universe_path)
    return local_client.get_history(symbol, period=period, interval=interval)


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
    df["AvgVolume20"] = df["Volume"].rolling(20).mean()

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
    fig.add_trace(go.Scatter(x=df["Date"], y=df["AvgVolume20"], mode="lines", name="Avg Volume 20"))
    fig.update_layout(
        title="Volume Trend",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=320,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig


def _build_normalized_compare(compare_frames: dict[str, pd.DataFrame]):
    merged = None

    for label, df in compare_frames.items():
        clean = _standardize_history(df)
        if clean.empty:
            continue

        base = _num(clean["Close"].iloc[0])
        if base in (None, 0):
            continue

        norm = clean[["Date", "Close"]].copy()
        norm[label] = (norm["Close"] / base) * 100
        norm = norm[["Date", label]]

        if merged is None:
            merged = norm
        else:
            merged = pd.merge(merged, norm, on="Date", how="outer")

    if merged is None or merged.empty:
        return None

    merged = merged.sort_values("Date")
    fig = go.Figure()
    for col in merged.columns:
        if col == "Date":
            continue
        fig.add_trace(go.Scatter(x=merged["Date"], y=merged[col], mode="lines", name=col))

    fig.update_layout(
        title="Normalized Performance Comparison (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalized Return",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h"),
    )
    return fig

# send_to_analyst_workspace imported from src.app_state


universe_df = load_universe_cached(str(UNIVERSE_PATH))

with st.expander("Universe debug"):
    st.write(f"Universe path: {UNIVERSE_PATH}")
    st.write(f"Universe rows: {len(universe_df)}")
    if not universe_df.empty:
        st.dataframe(universe_df, use_container_width=True, hide_index=True)

tabs = st.tabs(["Single Stock", "Watchlist", "Symbol Helper"])

with tabs[0]:
    st.subheader("Single Stock Lookup")

    search_col, symbol_col = st.columns([2, 1])

    search_text = search_col.text_input(
        "Search company or symbol from full CSE universe",
        placeholder="e.g. John Keells, Commercial Bank, JKH, COMB",
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

    selected_label = search_col.selectbox(
        "Pick from full company universe",
        options=[""] + list(option_map.keys()),
        index=0,
    )

    manual_symbol = symbol_col.text_input(
        "Or type symbol / alias",
        placeholder="e.g. JKH, JKH.N, JKH.N0000",
    )

    typed_symbol = client.normalize_symbol_text(manual_symbol)
    selected_universe_symbol = option_map.get(selected_label, "")
    final_symbol = (
        client.resolve_symbol_from_universe(typed_symbol, universe_df)
        if typed_symbol
        else selected_universe_symbol
    )

    chart_col1, chart_col2 = st.columns(2)
    period = chart_col1.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = chart_col2.selectbox("History Interval", ["1d", "1wk", "1mo"], index=0)

    if st.button("Lookup Symbol", use_container_width=True):
        if not final_symbol:
            st.warning("Pick a company from the full universe or type a symbol.")
        else:
            try:
                quote = client.get_quote(final_symbol)
                set_active_symbol(quote.canonical_symbol, quote.company_name or quote.canonical_symbol)
                hist_raw = load_history_cached(final_symbol, str(UNIVERSE_PATH), period, interval)
                hist = _standardize_history(hist_raw)

                st.markdown(f"### {quote.company_name or quote.canonical_symbol}")
                st.caption(f"Requested symbol: {manual_symbol.strip().upper() or final_symbol}")
                st.caption(f"Resolved CSE symbol: {quote.canonical_symbol}")
                st.caption(f"Yahoo Finance symbol: {quote.yahoo_symbol}")
                if quote.currency:
                    st.caption(f"Currency: {quote.currency}")

                action_col1, action_col2 = st.columns([1, 1])
                if action_col1.button("Send to Analyst Workspace", use_container_width=True):
                    send_to_analyst_workspace(
                        company_name=quote.company_name or quote.canonical_symbol,
                        ticker=quote.canonical_symbol,
                        query=f"What are the latest investment implications, catalysts, and risks for {quote.company_name or quote.canonical_symbol}?",
                    )
                if action_col2.button("Open Stock Research", use_container_width=True):
                    send_to_stock_research(quote.canonical_symbol, quote.company_name or quote.canonical_symbol)

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                metric_col1.metric(
                    "Last Traded Price",
                    _fmt_num(quote.last_traded_price),
                    _fmt_delta(quote.change, quote.change_pct),
                )
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

                if not hist.empty:
                    ret_col1, ret_col2, ret_col3, ret_col4, ret_col5 = st.columns(5)
                    ret_col1.metric("1W Return", _fmt_delta(None, _return_from_days(hist, 7)))
                    ret_col2.metric("1M Return", _fmt_delta(None, _return_from_days(hist, 30)))
                    ret_col3.metric("3M Return", _fmt_delta(None, _return_from_days(hist, 90)))
                    ret_col4.metric("6M Return", _fmt_delta(None, _return_from_days(hist, 180)))
                    ret_col5.metric("YTD Return", _fmt_delta(None, _ytd_return(hist)))

                    st.plotly_chart(
                        _build_price_chart(hist, f"{quote.company_name or quote.canonical_symbol} Price History"),
                        use_container_width=True,
                    )
                    st.plotly_chart(_build_volume_chart(hist), use_container_width=True)

                    with st.expander("Show historical data table"):
                        st.dataframe(hist, use_container_width=True, hide_index=True)
                else:
                    st.warning("No historical price series returned from Yahoo Finance for this symbol.")

                with st.expander("Raw Yahoo info"):
                    st.json(quote.raw_info or {})

            except Exception as e:
                st.error(f"Failed to fetch Yahoo Finance data: {e}")

with tabs[1]:
    st.subheader("Watchlist")

    watchlist_text = st.text_area(
        "Enter one symbol per line",
        placeholder="JKH\nSAMP\nCOMB.N0000",
        height=140,
    )

    compare_period = st.selectbox("Comparison Period", ["1mo", "3mo", "6mo", "1y"], index=2, key="compare_period")

    if st.button("Load Watchlist", use_container_width=True):
        symbols = [line.strip().upper() for line in watchlist_text.splitlines() if line.strip()]
        if not symbols:
            st.warning("Enter at least one symbol.")
        else:
            try:
                df = client.get_watchlist_quotes(symbols)
                if df.empty:
                    st.warning("No watchlist data returned.")
                else:
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    compare_frames = {}
                    for _, row in df.iterrows():
                        canonical_symbol = row["canonical_symbol"]
                        label = row["company_name"] or canonical_symbol
                        hist = load_history_cached(canonical_symbol, str(UNIVERSE_PATH), compare_period, "1d")
                        compare_frames[label] = hist

                    fig = _build_normalized_compare(compare_frames)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Failed to load watchlist: {e}")

with tabs[2]:
    st.subheader("Symbol Helper")
    st.caption("Use this to verify how your CSE symbol maps to Yahoo Finance.")

    helper_symbol = st.text_input(
        "Type a symbol alias",
        placeholder="e.g. JKH, JKH.N, JKH.N0000",
        key="helper_symbol",
    )

    if helper_symbol.strip():
        resolved = client.resolve_symbol_from_universe(helper_symbol, universe_df)
        yahoo_symbol = client.cse_to_yahoo_symbol(resolved)

        st.write("Input:", helper_symbol.strip().upper())
        st.write("Resolved CSE symbol:", resolved)
        st.write("Yahoo symbol:", yahoo_symbol)