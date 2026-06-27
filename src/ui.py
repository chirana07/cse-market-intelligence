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


def _escape(value: Any) -> str:
    return html.escape(str(value or ""))


def inject_global_styles() -> None:
    st.html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{_FONT_URL}" rel="stylesheet">
        <style>
        :root {{
            --bg: #f5f7fb;
            --panel: #ffffff;
            --panel-muted: #f8fafc;
            --border: #d9e0ea;
            --border-strong: #cbd5e1;
            --text: #17202a;
            --muted: #64748b;
            --subtle: #94a3b8;
            --accent: #2563eb;
            --accent-soft: #dbeafe;
            --success: #15803d;
            --warning: #b45309;
            --danger: #b91c1c;
            --radius: 8px;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--text);
        }}

        .stApp {{
            background: var(--bg);
        }}

        .block-container {{
            max-width: 1440px;
            padding-top: 1.1rem;
            padding-bottom: 3rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            letter-spacing: 0;
            color: var(--text);
        }}
        h1 {{ font-size: 1.7rem; font-weight: 700; }}
        h2 {{ font-size: 1.3rem; font-weight: 700; }}
        h3 {{ font-size: 1rem; font-weight: 650; }}
        h4 {{ font-size: 0.95rem; font-weight: 650; }}
        p, li, label, span {{
            letter-spacing: 0;
        }}

        header[data-testid="stHeader"] {{
            background: rgba(245,247,251,0.92);
            border-bottom: 1px solid rgba(203,213,225,0.85);
        }}

        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid var(--border);
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text);
        }}

        div[data-testid="stMetric"] {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.8rem 0.9rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.74rem;
            font-weight: 650;
            color: var(--muted);
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text);
        }}
        div[data-testid="stMetricDelta"] {{
            font-size: 0.8rem;
            font-weight: 600;
        }}

        div.stButton > button,
        div.stDownloadButton > button,
        div.stLinkButton > a {{
            border-radius: var(--radius);
            border: 1px solid var(--border-strong);
            background: #ffffff;
            color: var(--text);
            font-weight: 650;
            font-size: 0.86rem;
            letter-spacing: 0;
            padding: 0.45rem 0.8rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div.stLinkButton > a:hover {{
            border-color: var(--accent);
            color: var(--accent);
            background: #f8fbff;
        }}
        div.stButton > button[kind="primary"] {{
            background: var(--accent);
            border-color: var(--accent);
            color: #ffffff;
        }}
        div.stButton > button[kind="primary"]:hover {{
            background: #1d4ed8;
            color: #ffffff;
        }}

        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stNumberInput"] input {{
            border-radius: var(--radius);
            border: 1px solid var(--border-strong);
            background: #ffffff;
            color: var(--text);
            font-size: 0.9rem;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 1px var(--accent);
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            border-radius: var(--radius);
            border-color: var(--border-strong);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        div[data-testid="stDataFrame"] table {{
            font-size: 0.86rem;
        }}

        div[data-testid="stExpander"] {{
            border-radius: var(--radius);
            border: 1px solid var(--border);
            background: var(--panel);
        }}
        div[data-testid="stExpander"] summary {{
            font-weight: 650;
            font-size: 0.88rem;
            color: var(--text);
        }}

        div[role="tablist"] {{
            gap: 4px;
            padding: 3px;
            background: #eef2f7;
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
            background: #ffffff;
            color: var(--text);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
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
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}

        .cc-hero {{
            background: #ffffff;
            border: 1px solid var(--border);
            border-left: 4px solid var(--accent);
            border-radius: var(--radius);
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .cc-hero-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }}
        .cc-hero-title {{
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.25;
            color: var(--text);
            margin-bottom: 0.25rem;
        }}
        .cc-hero-caption {{
            font-size: 0.92rem;
            color: var(--muted);
            max-width: 760px;
        }}

        .cc-section-header {{
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--muted);
            margin: 0.2rem 0 0.65rem 0;
        }}

        .cc-card {{
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.9rem 1rem;
            background: var(--panel);
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}
        .cc-card:hover {{
            border-color: var(--border-strong);
        }}
        .cc-kicker {{
            font-size: 0.76rem;
            font-weight: 650;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }}
        .cc-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.2rem;
            line-height: 1.35;
        }}
        .cc-subtle,
        .cc-card-meta {{
            color: var(--muted);
            font-size: 0.86rem;
        }}

        .cc-badge {{
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0;
            vertical-align: middle;
            margin-right: 0.25rem;
            border: 1px solid transparent;
        }}
        .cc-chip {{
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--muted);
            background: #f1f5f9;
            border: 1px solid var(--border);
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
        }}

        .cc-empty {{
            text-align: center;
            padding: 2rem 1rem;
            color: var(--muted);
            background: var(--panel);
            border: 1px dashed var(--border-strong);
            border-radius: var(--radius);
        }}
        .cc-empty-title {{
            font-size: 0.98rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.25rem;
        }}
        .cc-empty-msg {{
            font-size: 0.88rem;
            color: var(--muted);
            max-width: 420px;
            margin: 0 auto;
        }}

        .chat-user,
        .chat-assistant {{
            border-radius: var(--radius);
            padding: 0.85rem 1rem;
            margin: 0.65rem 0;
            font-size: 0.93rem;
            line-height: 1.55;
        }}
        .chat-user {{
            background: var(--accent-soft);
            border: 1px solid #bfdbfe;
            max-width: 82%;
            margin-left: auto;
        }}
        .chat-assistant {{
            background: var(--panel);
            border: 1px solid var(--border);
            max-width: 100%;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
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
            background: #ffffff;
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
        ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 8px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
        </style>
        """,
    )


def page_header(title: str, caption: str = "", badge: str = "") -> None:
    badge_html = ""
    if badge:
        badge_html = status_badge(badge, "info")
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
