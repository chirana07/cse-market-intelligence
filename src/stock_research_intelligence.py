from __future__ import annotations

from langchain_ollama import ChatOllama

from src.config import CHAT_MODEL, OLLAMA_BASE_URL


def _get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def generate_stock_ai_view(
    company_name: str,
    ticker: str,
    quote_snapshot: str,
    market_stats: str,
    latest_announcement_summary: str,
    latest_report_summary: str,
) -> str:
    llm = _get_llm()

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Use only the provided information.
Do not invent facts.
If a section has weak evidence, say so clearly.

Return the answer in this exact structure:

1. Investment Thesis
- 3 to 6 bullet points

2. Bullish Signals
- Bullet points

3. Bearish Signals
- Bullet points

4. Catalysts
- Bullet points

5. Risks
- Bullet points

6. What To Monitor Next
- Bullet points

Company: {company_name or "Not specified"}
Ticker: {ticker or "Not specified"}

Quote Snapshot:
{quote_snapshot}

Market Stats:
{market_stats}

Latest Announcement Summary:
{latest_announcement_summary or "No announcement summary available."}

Latest Report Summary:
{latest_report_summary or "No report summary available."}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)