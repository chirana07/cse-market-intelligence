from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

import requests
from langchain_ollama import ChatOllama
from pypdf import PdfReader

from src.config import CHAT_MODEL, OLLAMA_BASE_URL


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_pdf_text_from_bytes(pdf_bytes: bytes, max_pages: int = 50, max_chars: int = 40000) -> str:
    if not pdf_bytes:
        return ""

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception:
        return ""

    chunks = []
    for page in reader.pages[:max_pages]:
        try:
            chunks.append(page.extract_text() or "")
        except Exception:
            continue

    text = "\n".join(chunks)
    text = _clean_text(text)
    return text[:max_chars]


def extract_pdf_text_from_file(path: str | Path, max_pages: int = 50, max_chars: int = 40000) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return extract_pdf_text_from_bytes(path.read_bytes(), max_pages=max_pages, max_chars=max_chars)


def extract_pdf_text_from_url(url: str, timeout: int = 30, max_pages: int = 50, max_chars: int = 40000) -> str:
    if not url:
        return ""

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cse.lk/",
        },
    )
    response.raise_for_status()

    return extract_pdf_text_from_bytes(
        response.content,
        max_pages=max_pages,
        max_chars=max_chars,
    )


def _get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def summarize_report(
    company_name: str,
    ticker: str,
    report_type: str,
    report_text: str,
) -> str:
    llm = _get_llm()
    clipped = (report_text or "")[:22000]

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Analyze the following company report.
Use only the provided text.
Do not invent facts or numbers.
If the evidence is incomplete, say so clearly.

Return the answer in this exact structure:

1. Executive Summary
- 4 to 8 bullet points

2. Financial Highlights
- Mention revenue, profit, margins, balance sheet, cash flow, dividends, if present

3. Management / Strategy Signals
- Bullet points

4. Risks / Warning Signs
- Bullet points

5. Positive Signals
- Bullet points

6. Outlook
- Explain what management seems to expect next

7. Investor Questions
- Bullet points

Company: {company_name or "Not specified"}
Ticker: {ticker or "Not specified"}
Report Type: {report_type}

Report Text:
{clipped}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


def compare_reports(
    company_name: str,
    ticker: str,
    latest_label: str,
    latest_text: str,
    previous_label: str,
    previous_text: str,
) -> str:
    llm = _get_llm()

    latest_clipped = (latest_text or "")[:18000]
    previous_clipped = (previous_text or "")[:18000]

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Compare the latest company report against the previous report.
Use only the provided text.
Do not invent facts or numbers.
If the evidence is incomplete, say so clearly.

Return the answer in this exact structure:

1. What Improved
- Bullet points

2. What Weakened
- Bullet points

3. What Stayed Similar
- Bullet points

4. Management Tone Shift
- Explain whether the tone became more positive, negative, or unchanged

5. Investor Relevance
- Explain why these changes matter

6. Follow-up Questions
- Bullet points

Company: {company_name or "Not specified"}
Ticker: {ticker or "Not specified"}

Latest Report Label: {latest_label}
Latest Report Text:
{latest_clipped}

Previous Report Label: {previous_label}
Previous Report Text:
{previous_clipped}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)