import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────
# NAVIGATION — must be called before any other st calls
# ─────────────────────────────────────────────────────────
pg = st.navigation(
    {
        "Command": [
            st.Page("src/views/command_center.py", title="Command Center", icon=":material/dashboard:"),
        ],
        "Research": [
            st.Page("src/views/stock_research.py",       title="Stock Research",       icon=":material/stacked_line_chart:"),
            st.Page("src/views/announcements_hub.py",    title="Announcements Hub",    icon=":material/campaign:"),
            st.Page("src/views/report_intelligence.py", title="Report Intelligence",   icon=":material/quick_reference_all:"),
            st.Page("src/views/analyst_workspace.py",                          title="AI Analyst Copilot",   icon=":material/robot_2:"),
        ],
        "Portfolio": [
            st.Page("src/views/portfolio_intelligence.py", title="Portfolio Intelligence", icon=":material/pie_chart:"),
            st.Page("src/views/stock_screener.py",           title="Stock Screener",         icon=":material/filter_alt:"),
        ],
        "Monitoring": [
            st.Page("src/views/alerts_monitoring.py",  title="Alerts & Monitoring",   icon=":material/notifications_active:"),
            st.Page("src/views/market_dashboard.py",   title="Market Dashboard",      icon=":material/monitoring:"),
        ],
    }
)
pg.run()