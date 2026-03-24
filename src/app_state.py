from __future__ import annotations

import streamlit as st


def set_active_symbol(symbol: str, company_name: str = "") -> None:
    st.session_state.active_symbol = (symbol or "").strip().upper()
    st.session_state.active_company_name = (company_name or "").strip()


def get_active_symbol() -> str:
    return (st.session_state.get("active_symbol", "") or "").strip().upper()


def get_active_company_name() -> str:
    return (st.session_state.get("active_company_name", "") or "").strip()


def send_to_stock_research(symbol: str, company_name: str = "") -> None:
    clean_symbol = (symbol or "").strip().upper()
    set_active_symbol(clean_symbol, company_name)
    st.session_state.pending_stock_research_symbol = clean_symbol
    st.switch_page("src/views/stock_research.py")


def send_to_analyst_workspace(
    company_name: str,
    ticker: str,
    analysis_mode: str = "News Summary",
    query: str = "",
) -> None:
    clean_ticker = (ticker or "").strip().upper()
    set_active_symbol(clean_ticker, company_name)
    st.session_state.pending_market_selection = {
        "company_name": company_name or "",
        "ticker": clean_ticker,
        "analysis_mode": analysis_mode or "News Summary",
        "query": query or f"Analyze {company_name or clean_ticker}.",
    }
    st.switch_page("src/views/analyst_workspace.py")


def send_to_announcements(company_name: str, ticker: str = "") -> None:
    clean_ticker = (ticker or "").strip().upper()
    set_active_symbol(clean_ticker, company_name)
    st.session_state.timeline_company = company_name or ""
    st.switch_page("src/views/announcements_hub.py")


def send_to_portfolio_review(symbols: list[str]) -> None:
    clean_symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
    st.session_state.pending_portfolio_symbols = clean_symbols
    st.switch_page("src/views/portfolio_intelligence.py")