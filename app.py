import streamlit as st


# ─────────────────────────────────────────────────────────
# NAVIGATION — must be called before any other st calls
# ─────────────────────────────────────────────────────────
pg = st.navigation(
    [
        st.Page("src/views/command_center.py", title="Command Center", icon=":material/dashboard:"),
        st.Page("src/views/stock_research.py", title="Stock Research", icon=":material/stacked_line_chart:"),
        st.Page("src/views/report_intelligence.py", title="Document Intelligence", icon=":material/quick_reference_all:"),
        st.Page("src/views/portfolio_intelligence.py", title="Portfolio & Screener", icon=":material/pie_chart:"),
        st.Page("src/views/analyst_workspace.py", title="Copilot & Monitoring", icon=":material/robot_2:"),
    ]
)
pg.run()
