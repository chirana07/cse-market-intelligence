from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────
_FONT_URL = "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"

_BADGE_COLORS = {
    "high":       ("rgba(239,68,68,0.15)",   "#f87171"),
    "medium":     ("rgba(245,158,11,0.15)",  "#fbbf24"),
    "low":        ("rgba(100,116,139,0.1)",  "#94a3b8"),
    "positive":   ("rgba(34,197,94,0.12)",   "#4ade80"),
    "neutral":    ("rgba(100,116,139,0.1)",  "#94a3b8"),
    "cautious":   ("rgba(245,158,11,0.12)",  "#fbbf24"),
    "info":       ("rgba(99,102,241,0.12)",  "#a5b4fc"),
    "triggered":  ("rgba(239,68,68,0.15)",   "#f87171"),
}


# ─────────────────────────────────────────────────────────
#  GLOBAL STYLES — injected once per page
# ─────────────────────────────────────────────────────────
def inject_global_styles() -> None:
    st.html(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="{_FONT_URL}" rel="stylesheet">
        <style>

        /* ── BASE ── */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }}

        /* ── TYPOGRAPHY ── */
        h1 {{ font-size: 1.75rem; font-weight: 700; letter-spacing: -0.025em; }}
        h2 {{ font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }}
        h3 {{ font-size: 1.1rem;  font-weight: 600; letter-spacing: -0.015em; }}
        h4 {{ font-size: 0.95rem; font-weight: 600; }}

        /* ── METRIC CARDS ── */
        div[data-testid="stMetric"] {{
            background: linear-gradient(145deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            transition: border-color 0.2s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            border-color: rgba(255,255,255,0.14);
        }}
        div[data-testid="stMetricLabel"] {{
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            opacity: 0.65;
        }}
        div[data-testid="stMetricValue"] {{
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}
        div[data-testid="stMetricDelta"] {{
            font-size: 0.82rem;
            font-weight: 600;
        }}

        /* ── BUTTONS ── */
        div.stButton > button,
        div.stDownloadButton > button {{
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.01em;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.18s ease;
            padding: 0.45rem 0.9rem;
        }}
        div.stButton > button:hover {{
            border-color: rgba(255,255,255,0.22);
            background: rgba(255,255,255,0.06);
        }}
        div.stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            border: none;
        }}

        /* ── LINK BUTTONS ── */
        div.stLinkButton > a {{
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.875rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        /* ── DATAFRAMES / TABLES ── */
        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.07);
        }}
        div[data-testid="stDataFrame"] table {{
            font-size: 0.875rem;
        }}

        /* ── EXPANDERS ── */
        div[data-testid="stExpander"] {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.07);
            overflow: hidden;
        }}
        div[data-testid="stExpander"] summary {{
            font-weight: 600;
            font-size: 0.9rem;
        }}

        /* ── TABS ── */
        div[role="tablist"] {{
            gap: 2px;
            padding: 3px;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.07);
        }}
        div[role="tablist"] button {{
            border-radius: 9px;
            font-weight: 600;
            font-size: 0.875rem;
            padding: 0.4rem 0.85rem;
        }}
        div[role="tablist"] button[aria-selected="true"] {{
            background: rgba(255,255,255,0.08);
        }}

        /* ── SIDEBAR ── */
        section[data-testid="stSidebar"] {{
            border-right: 1px solid rgba(255,255,255,0.07);
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1rem;
        }}

        /* ── TEXT INPUTS ── */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextArea"] textarea {{
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
            border-radius: 10px;
        }}

        /* ── CONTAINERS / BORDER CARDS ── */
        div[data-testid="stVerticalBlock"] div[data-testid="element-container"] > div[style*="border"] {{
            border-radius: 16px !important;
        }}

        /* ── ALERTS / INFO BOXES ── */
        div[data-testid="stAlert"] {{
            border-radius: 12px;
            font-size: 0.9rem;
        }}

        /* ── CUSTOM COMPONENT CLASSES ── */

        /* Hero header */
        .cc-hero {{
            padding: 1.75rem 0 1.25rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.07);
            margin-bottom: 1.5rem;
        }}
        .cc-hero-eyebrow {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #6366f1;
            margin-bottom: 0.4rem;
        }}
        .cc-hero-title {{
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin-bottom: 0.45rem;
        }}
        .cc-hero-caption {{
            font-size: 0.95rem;
            opacity: 0.65;
            font-weight: 400;
            max-width: 640px;
        }}

        /* Section header */
        .cc-section-header {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            opacity: 0.55;
            margin-bottom: 0.75rem;
            margin-top: 0.25rem;
        }}

        /* Generic card */
        .cc-card {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.01) 100%);
            margin-bottom: 0.65rem;
            transition: border-color 0.18s ease;
        }}
        .cc-card:hover {{
            border-color: rgba(255,255,255,0.14);
        }}

        /* Card sub-elements */
        .cc-kicker {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.6;
            margin-bottom: 0.2rem;
        }}
        .cc-title {{
            font-size: 0.97rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            line-height: 1.35;
        }}
        .cc-subtle {{
            opacity: 0.75;
            font-size: 0.875rem;
        }}
        .cc-card-meta {{
            font-size: 0.78rem;
            opacity: 0.55;
            margin-top: 0.4rem;
        }}

        /* Status badges */
        .cc-badge {{
            display: inline-block;
            padding: 0.25em 0.85em;
            border-radius: 99px;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            vertical-align: middle;
            margin-right: 0.3rem;
            border: 1px solid rgba(255,255,255,0.05);
        }}

        /* Chip row */
        .cc-chip {{
            display: inline-block;
            padding: 0.2em 0.7em;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.1);
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
        }}

        /* Empty state */
        .cc-empty {{
            text-align: center;
            padding: 3rem 1rem;
            opacity: 0.7;
        }}
        .cc-empty-icon {{
            font-size: 2.5rem;
            margin-bottom: 0.75rem;
            display: block;
        }}
        .cc-empty-title {{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .cc-empty-msg {{
            font-size: 0.9rem;
            opacity: 0.7;
            max-width: 340px;
            margin: 0 auto 1rem auto;
        }}

        /* Chat bubbles */
        .chat-user {{
            background: rgba(99,102,241,0.14);
            border: 1px solid rgba(99,102,241,0.25);
            border-radius: 14px 14px 4px 14px;
            padding: 0.75rem 1rem;
            margin: 0.75rem 0;
            font-size: 0.92rem;
            max-width: 85%;
            margin-left: auto;
        }}
        .chat-assistant {{
            background: linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px 14px 14px 4px;
            padding: 1rem 1.15rem;
            margin: 0.75rem 0;
            font-size: 0.93rem;
            max-width: 95%;
        }}
        .chat-label-user {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #818cf8;
            margin-bottom: 0.3rem;
            text-align: right;
        }}
        .chat-label-assistant {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 0.3rem;
        }}

        /* Context banner */
        .cc-context-bar {{
            background: rgba(99,102,241,0.1);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-size: 0.88rem;
            font-weight: 500;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Divider label */
        .cc-divider {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin: 1.25rem 0 0.5rem 0;
            opacity: 0.4;
        }}
        .cc-divider-line {{
            flex: 1;
            height: 1px;
            background: rgba(255,255,255,0.15);
        }}
        .cc-divider-text {{
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            white-space: nowrap;
        }}

        /* Module launcher cards */
        .cc-nav-card {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            background: rgba(255,255,255,0.025);
            text-align: center;
            cursor: pointer;
            transition: all 0.18s ease;
            margin-bottom: 0.5rem;
        }}
        .cc-nav-card:hover {{
            border-color: rgba(99,102,241,0.4);
            background: rgba(99,102,241,0.06);
        }}
        .cc-nav-icon {{
            font-size: 1.4rem;
            display: block;
            margin-bottom: 0.2rem;
        }}
        .cc-nav-label {{
            font-size: 0.8rem;
            font-weight: 600;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.15); border-radius: 3px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.25); }}

        </style>
        """,
    )


# ─────────────────────────────────────────────────────────
#  COMPONENT HELPERS
# ─────────────────────────────────────────────────────────

def page_header(title: str, caption: str = "", badge: str = "") -> None:
    """Premium hero-style page header."""
    badge_html = ""
    if badge:
        badge_html = f'<span class="cc-badge" style="background:rgba(99,102,241,0.15);color:#a5b4fc;">{badge}</span>'
    st.markdown(
        f"""
        <div class="cc-hero">
            <div class="cc-hero-eyebrow">CSE Research Platform</div>
            <div class="cc-hero-title">{badge_html} {title}</div>
            {"<div class='cc-hero-caption'>" + caption + "</div>" if caption else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "") -> None:
    """Compact uppercase section label above a content block."""
    st.markdown(
        f'<div class="cc-section-header">{title}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


def metric_row(metrics: list[dict[str, any]]) -> None:
    """Render a row of premium metrics using st.columns."""
    if not metrics:
        return
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        cols[i].metric(
            label=m.get("label", ""),
            value=m.get("value", ""),
            delta=m.get("delta", None),
            delta_color=m.get("delta_color", "normal")
        )


def info_card(kicker: str, title: str, subtitle: str = "", tag: str = "", tag_level: str = "low") -> None:
    """A styled disclosure/alert card."""
    bg, color = _BADGE_COLORS.get(tag_level.lower(), _BADGE_COLORS["low"])
    tag_html = (
        f'<span class="cc-badge" style="background:{bg};color:{color};">{tag}</span>'
        if tag else ""
    )
    st.markdown(
        f"""
        <div class="cc-card">
            <div class="cc-kicker">{kicker} {tag_html}</div>
            <div class="cc-title">{title}</div>
            {"<div class='cc-subtle'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(label: str, level: str = "low") -> str:
    """Return an inline HTML badge string. Use with unsafe_allow_html=True."""
    bg, color = _BADGE_COLORS.get(level.lower(), _BADGE_COLORS["low"])
    return f'<span class="cc-badge" style="background:{bg};color:{color};">{label}</span>'


def chip_row(chips: list[str]) -> None:
    """Render a horizontal row of subtle chips."""
    if not chips:
        return
    html = " ".join(f'<span class="cc-chip">{c}</span>' for c in chips)
    st.markdown(html, unsafe_allow_html=True)


def empty_state(icon: str, title: str, message: str = "") -> None:
    """Centered empty-state block. Icon argument is ignored to enforce clean typography."""
    st.markdown(
        f"""
        <div class="cc-empty">
            <div class="cc-empty-title">{title}</div>
            {"<div class='cc-empty-msg'>" + message + "</div>" if message else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def context_bar(symbol: str, company: str) -> None:
    """Show the active research context as a styled banner."""
    if not symbol and not company:
        return
    text = f"<strong>{company or symbol}</strong>"
    if symbol and company:
        text = f"<strong>{company}</strong> <span style='opacity:0.6;font-size:0.85em;'>({symbol})</span>"
    st.markdown(
        f'<div class="cc-context-bar"><span style="opacity:0.4; margin-right:4px;">|</span> {text}</div>',
        unsafe_allow_html=True,
    )


def divider_label(text: str) -> None:
    """Horizontal rule with a centered label."""
    st.markdown(
        f"""
        <div class="cc-divider">
            <div class="cc-divider-line"></div>
            <span class="cc-divider-text">{text}</span>
            <div class="cc-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chat_message(role: str, content: str) -> None:
    """Render a chat bubble. role = 'user' or 'assistant'."""
    if role == "user":
        st.markdown(
            f"""
            <div class="chat-label-user">You</div>
            <div class="chat-user">{content}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="chat-label-assistant">AI Analyst</div>
            <div class="chat-assistant">{content}</div>
            """,
            unsafe_allow_html=True,
        )