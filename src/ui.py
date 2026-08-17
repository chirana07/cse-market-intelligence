from __future__ import annotations

import html
from typing import Any

import streamlit as st

_FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

_BADGE_COLORS = {
    "high": ("#fee2e2", "#991b1b", "#fecaca"),
    "medium": ("#fef3c7", "#92400e", "#fde68a"),
    "low": ("#f1f5f9", "#475569", "#e2e8f0"),
    "positive": ("#dcfce7", "#166534", "#bbf7d0"),
    "neutral": ("#f1f5f9", "#475569", "#e2e8f0"),
    "cautious": ("#fef3c7", "#92400e", "#fde68a"),
    "info": ("#dbeafe", "#1d4ed8", "#bfdbfe"),
    "triggered": ("#fee2e2", "#991b1b", "#fecaca"),
    "priority": ("#fee2e2", "#991b1b", "#fecaca"),
    "watch": ("#fef3c7", "#92400e", "#fde68a"),
    "routine": ("#f1f5f9", "#475569", "#e2e8f0"),
}

_THEME_TOKENS = {
    "Light": {
        "bg": "#f8fafc",
        "panel": "#ffffff",
        "panel_muted": "#f1f5f9",
        "border": "#e2e8f0",
        "border_strong": "#cbd5e1",
        "text": "#0f172a",
        "muted": "#64748b",
        "subtle": "#94a3b8",
        "accent": "#2563eb",
        "accent_hover": "#1d4ed8",
        "accent_soft": "#eff6ff",
        "button_hover": "#f8fafc",
        "tab_bg": "#f1f5f9",
        "shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.05)",
        "header_bg": "rgba(255, 255, 255, 0.92)",
        "sidebar_active": "#eff6ff",
    },
    "Dark": {
        "bg": "#0b0c0e",
        "panel": "#14171d",
        "panel_muted": "#1b1f27",
        "border": "#262b35",
        "border_strong": "#363d4b",
        "text": "#f8fafc",
        "muted": "#94a3b8",
        "subtle": "#64748b",
        "accent": "#3b82f6",
        "accent_hover": "#60a5fa",
        "accent_soft": "#1e293b",
        "button_hover": "#1e2430",
        "tab_bg": "#14171d",
        "shadow": "0 1px 3px 0 rgba(0, 0, 0, 0.3)",
        "header_bg": "rgba(20, 23, 29, 0.92)",
        "sidebar_active": "#1e293b",
    },
}


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def _selected_theme() -> str:
    current = st.session_state.get("ui_theme_mode", "Light")
    if current not in _THEME_TOKENS:
        current = "Light"
    return current


def _theme_control(key: str = "ui_theme_mode") -> None:
    current = _selected_theme()
    final_key = key
    if final_key in st.session_state:
        final_key = f"{key}_{st.session_state.get('_theme_ctrl_counter', 0) + 1}"
        st.session_state["_theme_ctrl_counter"] = st.session_state.get("_theme_ctrl_counter", 0) + 1

    selected = st.selectbox(
        "Appearance",
        ["Light", "Dark"],
        index=0 if current == "Light" else 1,
        key=final_key,
    )
    if selected != current:
        st.session_state.ui_theme_mode = selected
        st.rerun()


def inject_global_styles() -> None:
    theme_name = _selected_theme()
    tokens = _THEME_TOKENS[theme_name]
    st.html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{_FONT_URL}" rel="stylesheet">
        <style>
        :root {{
            --bg: {tokens["bg"]};
            --panel: {tokens["panel"]};
            --panel-muted: {tokens["panel_muted"]};
            --border: {tokens["border"]};
            --border-strong: {tokens["border_strong"]};
            --text: {tokens["text"]};
            --muted: {tokens["muted"]};
            --subtle: {tokens["subtle"]};
            --accent: {tokens["accent"]};
            --accent-hover: {tokens["accent_hover"]};
            --accent-soft: {tokens["accent_soft"]};
            --button-hover: {tokens["button_hover"]};
            --tab-bg: {tokens["tab_bg"]};
            --shadow: {tokens["shadow"]};
            --header-bg: {tokens["header_bg"]};
            --sidebar-active: {tokens["sidebar_active"]};
            --radius: 6px;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-feature-settings: "tnum" 1, "zero" 1;
            color: var(--text) !important;
        }}
        .stApp {{
            background: var(--bg);
        }}
        .block-container {{
            max-width: 1400px;
            padding-top: 3.1rem !important;
            padding-bottom: 2.5rem;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span,
        label,
        summary span {{
            color: var(--text) !important;
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stCaption {{
            color: var(--muted) !important;
        }}

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        input,
        textarea {{
            color: var(--text) !important;
            background-color: var(--panel) !important;
        }}

        div[data-baseweb="popover"] * {{
            color: var(--text) !important;
            background-color: var(--panel) !important;
        }}

        table, table th, table td, div[data-testid="stDataFrame"] * {{
            color: var(--text) !important;
        }}

        table th {{
            background-color: var(--panel-muted) !important;
            color: var(--muted) !important;
        }}

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span {{
            color: var(--text) !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            letter-spacing: -0.01em;
            color: var(--text) !important;
        }}
        h1 {{ font-size: 1.65rem; font-weight: 700; }}
        h2 {{ font-size: 1.25rem; font-weight: 650; }}
        h3 {{ font-size: 0.98rem; font-weight: 650; }}
        h4 {{ font-size: 0.9rem; font-weight: 650; }}
        p, li, label, span {{
            letter-spacing: -0.005em;
        }}

        header[data-testid="stHeader"] {{
            background: var(--header-bg);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(8px);
        }}

        section[data-testid="stSidebar"] {{
            background: var(--panel);
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
        }}
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li div[aria-selected="true"] {{
            background: var(--sidebar-active);
            border-radius: var(--radius);
        }}

        div[data-testid="stMetric"] {{
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-muted) 100%);
            border: 1px solid var(--border);
            border-top: 3px solid var(--accent);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: transform 150ms ease, border-color 150ms ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.45rem;
            font-weight: 700;
            font-feature-settings: "tnum" 1;
            color: var(--text);
            margin-top: 0.1rem;
        }}
        div[data-testid="stMetricDelta"] {{
            font-size: 0.8rem;
            font-weight: 600;
            font-feature-settings: "tnum" 1;
        }}

        div[data-baseweb="tab-list"] {{
            gap: 6px;
            padding: 4px;
            background: var(--tab-bg);
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-bottom: 1rem;
        }}
        button[data-baseweb="tab"] {{
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.86rem !important;
            padding: 0.45rem 1rem !important;
            color: var(--muted) !important;
            transition: all 150ms ease;
        }}
        button[aria-selected="true"] {{
            background: var(--panel) !important;
            color: var(--text) !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            border: 1px solid var(--border-strong) !important;
        }}

        div.stButton > button,
        div.stDownloadButton > button,
        div.stLinkButton > a {{
            border-radius: 6px;
            border: 1px solid var(--border-strong);
            background: var(--panel);
            color: var(--text);
            font-weight: 600;
            font-size: 0.84rem;
            padding: 0.45rem 0.85rem;
            box-shadow: var(--shadow);
            transition: all 150ms ease;
        }}
        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div.stLinkButton > a:hover {{
            border-color: var(--accent);
            color: var(--accent);
            background: var(--button-hover);
            transform: translateY(-1px);
        }}
        div.stButton > button[kind="primary"] {{
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
            font-weight: 650;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: var(--accent-hover);
            color: #ffffff;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {{
            border-radius: 6px;
            border: 1px solid var(--border-strong);
            background: var(--panel);
            color: var(--text);
            font-size: 0.88rem;
            padding: 0.45rem 0.75rem;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-soft);
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            border-radius: 6px;
            border-color: var(--border-strong);
            background: var(--panel);
            color: var(--text);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }}
        div[data-testid="stDataFrame"] table {{
            font-size: 0.84rem;
            font-feature-settings: "tnum" 1;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            font-feature-settings: "tnum" 1;
            margin: 0.75rem 0;
        }}
        table th {{
            text-align: left;
            padding: 0.5rem 0.75rem;
            background: var(--panel-muted);
            border-bottom: 2px solid var(--border-strong);
            font-weight: 650;
            color: var(--muted);
        }}
        table td {{
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid var(--border);
        }}
        table tr:hover td {{
            background: var(--panel-muted);
        }}

        div[data-testid="stExpander"] {{
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: var(--panel);
        }}
        div[data-testid="stExpander"] summary {{
            font-weight: 600;
            font-size: 0.86rem;
            color: var(--text);
        }}

        div[role="tablist"] {{
            gap: 4px;
            padding: 3px;
            background: var(--tab-bg);
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }}
        div[role="tablist"] button {{
            border-radius: 6px;
            font-weight: 650;
            font-size: 0.86rem;
            color: var(--muted);
        }}
        div[role="tablist"] button[aria-selected="true"] {{
            background: var(--panel);
            color: var(--text);
            box-shadow: var(--shadow);
        }}

        div[data-testid="stAlert"] {{
            border-radius: var(--radius);
            border: 1px solid var(--border);
            font-size: 0.9rem;
        }}

        div[data-testid="stVerticalBlock"] div[data-testid="element-container"] > div[style*="border"] {{
            border-radius: var(--radius) !important;
            border-color: var(--border) !important;
            background: var(--panel) !important;
            box-shadow: var(--shadow);
        }}

        .cc-hero {{
            background: linear-gradient(180deg, var(--panel) 0%, var(--panel-muted) 100%);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
            margin-top: 0.25rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
        }}
        .cc-hero-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--accent);
            margin-bottom: 0.3rem;
        }}
        .cc-hero-title {{
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.25;
            letter-spacing: -0.015em;
            color: var(--text);
            margin-bottom: 0.35rem;
        }}
        .cc-hero-caption {{
            font-size: 0.9rem;
            color: var(--muted);
            line-height: 1.45;
            max-width: 780px;
        }}

        .cc-section-header {{
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--muted);
            margin: 0.4rem 0 0.7rem 0;
            padding-bottom: 0.2rem;
            border-bottom: 1px solid var(--border);
        }}

        .cc-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            background: var(--panel);
            margin-bottom: 0.75rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
            transition: all 150ms ease;
        }}
        .cc-card:hover {{
            border-color: var(--accent);
            box-shadow: 0 4px 8px rgba(0,0,0,0.06);
        }}
        .cc-kicker {{
            font-size: 0.74rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }}
        .cc-title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.35rem;
            line-height: 1.35;
        }}
        .cc-subtle,
        .cc-card-meta {{
            color: var(--muted);
            font-size: 0.85rem;
        }}

        .cc-badge {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            vertical-align: middle;
            margin-right: 0.35rem;
            border: 1px solid transparent;
        }}
        .cc-chip {{
            display: inline-flex;
            align-items: center;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--muted);
            background: var(--panel-muted);
            border: 1px solid var(--border);
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
        }}

        .cc-empty {{
            text-align: center;
            padding: 2rem 1.25rem;
            color: var(--muted);
            background: var(--panel-muted);
            border: 1px dashed var(--border-strong);
            border-radius: 8px;
        }}
        .cc-empty-title {{
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.35rem;
        }}
        .cc-empty-msg {{
            font-size: 0.88rem;
            color: var(--muted);
            max-width: 440px;
            margin: 0 auto;
        }}

        .chat-user,
        .chat-assistant {{
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            margin: 0.75rem 0;
            font-size: 0.92rem;
            line-height: 1.55;
        }}
        .chat-user {{
            background: var(--accent-soft);
            border: 1px solid var(--border-strong);
            max-width: 80%;
            margin-left: auto;
        }}
        .chat-assistant {{
            background: var(--panel);
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.03);
            max-width: 100%;
            box-shadow: var(--shadow);
        }}
        .chat-label-user,
        .chat-label-assistant {{
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }}
        .chat-label-user {{
            text-align: right;
        }}

        .cc-context-bar {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: var(--radius);
            padding: 0.58rem 0.85rem;
            font-size: 0.88rem;
            font-weight: 500;
            color: var(--text);
            margin-bottom: 0.9rem;
        }}

        .cc-divider {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 1.05rem 0 0.55rem 0;
        }}
        .cc-divider-line {{
            flex: 1;
            height: 1px;
            background: var(--border);
        }}
        .cc-divider-text {{
            font-size: 0.74rem;
            font-weight: 700;
            color: var(--muted);
            white-space: nowrap;
        }}

        .cc-nav-card {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.75rem 0.9rem;
            background: var(--panel);
            text-align: center;
            margin-bottom: 0.45rem;
        }}
        .cc-nav-card:hover {{
            border-color: var(--accent);
        }}
        .cc-nav-label {{
            font-size: 0.82rem;
            font-weight: 650;
            color: var(--text);
        }}

        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: var(--border-strong); border-radius: 8px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--subtle); }}

        @media (max-width: 640px) {{
            .block-container {{
                padding-top: 3.6rem !important;
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .cc-hero {{
                margin-top: 0.25rem;
                padding: 0.9rem 1rem;
            }}
            .cc-hero-title {{
                font-size: 1.3rem;
            }}
            div[data-testid="stMetric"] {{
                padding: 0.75rem 0.85rem;
            }}
            .cc-empty {{
                padding: 1.25rem 0.9rem;
            }}
        }}
        </style>
        """,
    )


def page_header(title: str, caption: str = "", badge: str = "") -> None:
    badge_html = status_badge(badge, "info") if badge else ""
    header_left, header_right = st.columns([6, 1.15])
    with header_left:
        st.markdown(
            f"""
            <div class="cc-hero">
                <div class="cc-hero-eyebrow">CSE Market Intelligence</div>
                <div class="cc-hero-title">{badge_html}{_escape(title)}</div>
                {"<div class='cc-hero-caption'>" + _escape(caption) + "</div>" if caption else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        _theme_control()


def section_header(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="cc-section-header">{_escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def metric_row(metrics: list[dict[str, Any]]) -> None:
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        cols[i].metric(
            label=metric.get("label", ""),
            value=metric.get("value", ""),
            delta=metric.get("delta", None),
            delta_color=metric.get("delta_color", "normal"),
        )


def info_card(kicker: str, title: str, subtitle: str = "", tag: str = "", tag_level: str = "low") -> None:
    tag_html = status_badge(tag, tag_level) if tag else ""
    st.markdown(
        f"""
        <div class="cc-card">
            <div class="cc-kicker">{_escape(kicker)} {tag_html}</div>
            <div class="cc-title">{_escape(title)}</div>
            {"<div class='cc-subtle'>" + _escape(subtitle) + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, level: str = "low") -> str:
    theme_name = _selected_theme()
    if theme_name == "Dark":
        dark_badges = {
            "high": ("#3f1212", "#fca5a5", "#7f1d1d"),
            "medium": ("#3a2007", "#fcd34d", "#78350f"),
            "low": ("#1e293b", "#cbd5e1", "#334155"),
            "positive": ("#062c19", "#86efac", "#14532d"),
            "neutral": ("#1e293b", "#cbd5e1", "#334155"),
            "cautious": ("#3a2007", "#fcd34d", "#78350f"),
            "info": ("#0c2a4a", "#93c5fd", "#1e3a8a"),
            "triggered": ("#3f1212", "#fca5a5", "#7f1d1d"),
            "priority": ("#3f1212", "#fca5a5", "#7f1d1d"),
            "watch": ("#3a2007", "#fcd34d", "#78350f"),
            "routine": ("#1e293b", "#cbd5e1", "#334155"),
        }
        bg, color, border = dark_badges.get(level.lower(), dark_badges["low"])
    else:
        bg, color, border = _BADGE_COLORS.get(level.lower(), _BADGE_COLORS["low"])
    return (
        f'<span class="cc-badge" '
        f'style="background:{bg};color:{color};border-color:{border};">{_escape(label)}</span>'
    )


def chip_row(chips: list[str]) -> None:
    if not chips:
        return
    html_row = " ".join(f'<span class="cc-chip">{_escape(chip)}</span>' for chip in chips)
    st.markdown(html_row, unsafe_allow_html=True)


def empty_state(icon: str, title: str, message: str = "") -> None:
    st.markdown(
        f"""
        <div class="cc-empty">
            <div class="cc-empty-title">{_escape(title)}</div>
            {"<div class='cc-empty-msg'>" + _escape(message) + "</div>" if message else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_bar(symbol: str, company: str) -> None:
    if not symbol and not company:
        return
    if symbol and company:
        text = f"<strong>{_escape(company)}</strong> <span style='color:var(--muted);'>({_escape(symbol)})</span>"
    else:
        text = f"<strong>{_escape(company or symbol)}</strong>"
    st.markdown(f'<div class="cc-context-bar">{text}</div>', unsafe_allow_html=True)


def divider_label(text: str) -> None:
    st.markdown(
        f"""
        <div class="cc-divider">
            <div class="cc-divider-line"></div>
            <span class="cc-divider-text">{_escape(text)}</span>
            <div class="cc-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_message(role: str, content: str) -> None:
    safe_content = _escape(content).replace("\n", "<br>")
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-label-user">You</div>
            <div class="chat-user">{safe_content}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-label-assistant">Analyst Copilot</div>
            <div class="chat-assistant">{safe_content}</div>
            """,
            unsafe_allow_html=True,
        )


def render_company_selector(universe_df: Any, key_prefix: str = "comp_sel") -> tuple[str, str]:
    """Unified, sleek company & ticker selection component for Streamlit views."""
    from src.app_state import set_active_symbol, get_active_symbol, get_active_company_name
    import pandas as pd

    curr_symbol = get_active_symbol()
    curr_company = get_active_company_name()

    if not isinstance(universe_df, pd.DataFrame) or universe_df.empty:
        col1, col2 = st.columns([2, 1])
        sym = col1.text_input("Ticker Symbol", value=curr_symbol, key=f"{key_prefix}_sym_input")
        comp = col2.text_input("Company Name", value=curr_company, key=f"{key_prefix}_comp_input")
        if sym:
            set_active_symbol(sym, comp)
        return sym.strip().upper(), comp.strip()

    col1, col2 = st.columns([2, 1])
    search_q = col1.text_input(
        "Search Company or Symbol",
        placeholder="e.g. John Keells, JKH, COMB, DIAL",
        key=f"{key_prefix}_search",
    )

    matches = universe_df.copy()
    if search_q.strip():
        q = search_q.strip().upper()
        matches = matches[
            matches["symbol"].astype(str).str.contains(q, na=False)
            | matches["company_name"].astype(str).str.upper().str.contains(q, na=False)
        ]

    option_map = {
        f"{row['company_name']} ({row['symbol']})": (str(row['symbol']), str(row['company_name']))
        for _, row in matches.head(100).iterrows()
    }

    options_list = [""] + list(option_map.keys())
    
    # Calculate initial index if active symbol matches an option
    default_idx = 0
    if curr_symbol:
        for idx, opt in enumerate(options_list[1:], start=1):
            sym_part, comp_part = option_map[opt]
            if sym_part == curr_symbol.upper():
                default_idx = idx
                break

    selected_opt = col2.selectbox(
        "Select Asset",
        options=options_list,
        index=default_idx,
        key=f"{key_prefix}_select",
    )

    if selected_opt and selected_opt in option_map:
        res_sym, res_comp = option_map[selected_opt]
    elif search_q.strip():
        res_sym = search_q.strip().upper()
        res_comp = ""
    else:
        res_sym = curr_symbol
        res_comp = curr_company

    if res_sym:
        set_active_symbol(res_sym, res_comp)

    return res_sym, res_comp

