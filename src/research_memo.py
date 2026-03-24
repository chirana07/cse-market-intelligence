from datetime import datetime, timezone
import re


def _safe_text(value: str, fallback: str = "Not specified") -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def build_memo_filename(company_name: str, ticker: str, analysis_mode: str) -> str:
    base = ticker or company_name or "equity_research"
    raw = f"{base}_{analysis_mode}_memo".lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return f"{slug}.md"


def build_research_memo_markdown(
    company_name: str,
    ticker: str,
    analysis_mode: str,
    user_query: str,
    answer: str,
    source_docs,
    selected_domain: str = "All",
    selected_source: str = "All",
    selected_ticker: str = "All",
    selected_event: str = "All",
):
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Equity Research Memo",
        "",
        f"**Created:** {created_at}",
        f"**Company:** {_safe_text(company_name)}",
        f"**Ticker:** {_safe_text(ticker)}",
        f"**Analysis Mode:** {_safe_text(analysis_mode)}",
        f"**Domain Filter:** {_safe_text(selected_domain)}",
        f"**Source Filter:** {_safe_text(selected_source)}",
        f"**Ticker Filter:** {_safe_text(selected_ticker)}",
        f"**Event Filter:** {_safe_text(selected_event)}",
        "",
        "## Research Question",
        "",
        user_query.strip() if user_query.strip() else "No question recorded.",
        "",
        "## Structured Analyst Output",
        "",
        answer.strip() if answer.strip() else "No answer generated.",
        "",
        "## Evidence Used",
        "",
    ]

    unique_sources = {}
    for doc in source_docs or []:
        source = doc.metadata.get("source", "Unknown source")
        if source in unique_sources:
            continue

        excerpt = " ".join((doc.page_content or "").split())
        excerpt = excerpt[:350] + ("..." if len(excerpt) > 350 else "")

        unique_sources[source] = {
            "title": doc.metadata.get("title", "Untitled"),
            "domain": doc.metadata.get("domain", "unknown"),
            "source": source,
            "tickers": doc.metadata.get("ticker_candidates_str", ""),
            "event_tags": doc.metadata.get("event_tags_str", ""),
            "excerpt": excerpt,
        }

    if not unique_sources:
        lines.append("- No source documents recorded.")
    else:
        for idx, item in enumerate(unique_sources.values(), start=1):
            lines.extend([
                f"### {idx}. {item['title']}",
                "",
                f"- **Domain:** {item['domain']}",
                f"- **Source URL:** {item['source']}",
                f"- **Detected Tickers:** {item['tickers'] or 'None detected'}",
                f"- **Detected Event Tags:** {item['event_tags'] or 'None detected'}",
                f"- **Excerpt:** {item['excerpt']}",
                "",
            ])

    return "\n".join(lines)