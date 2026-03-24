from __future__ import annotations

from io import BytesIO
import re

import requests
from bs4 import BeautifulSoup
from langchain_ollama import ChatOllama
from pypdf import PdfReader

from src.config import CHAT_MODEL, OLLAMA_BASE_URL


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def fetch_announcement_text(url: str, timeout: int = 30, max_chars: int = 25000) -> str:
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

    content_type = response.headers.get("Content-Type", "").lower()
    is_pdf = url.lower().endswith(".pdf") or "application/pdf" in content_type

    if is_pdf:
        try:
            reader = PdfReader(BytesIO(response.content))
            chunks = []
            for page in reader.pages[:30]:
                try:
                    chunks.append(page.extract_text() or "")
                except Exception:
                    continue
            text = "\n".join(chunks)
        except Exception:
            text = ""
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

    text = _clean_text(text)
    return text[:max_chars]


def _get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def summarize_announcement_text(
    company_name: str,
    title: str,
    category: str,
    text: str,
) -> str:
    llm = _get_llm()
    clipped_text = (text or "")[:18000]

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Analyze the following company announcement or disclosure.
Use only the provided text.
Do not invent facts.
If the text is incomplete, say so clearly.

Return the answer in this exact structure:

1. What Happened
- 3 to 6 bullet points

2. Why It Matters
- Explain the business, financial, governance, or market relevance

3. Investor Signals
- Bullish signals
- Bearish signals

4. Risks / Unknowns
- Bullet points

5. Follow-up Questions
- Bullet points

Company: {company_name}
Category: {category}
Announcement Title: {title}

Announcement Text:
{clipped_text}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)


def compare_announcements(
    company_name: str,
    latest_title: str,
    latest_text: str,
    previous_title: str,
    previous_text: str,
) -> str:
    llm = _get_llm()

    latest_clipped = (latest_text or "")[:12000]
    previous_clipped = (previous_text or "")[:12000]

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Compare the latest company disclosure against the previous disclosure.
Use only the provided text.
Do not invent facts.
If either text is incomplete, say so clearly.

Return the answer in this exact structure:

1. What Changed
- Bullet points of new facts or changes

2. What Stayed Similar
- Bullet points

3. Why The Delta Matters
- Explain investor relevance

4. New Risks / New Positives
- Bullet points

5. Questions For Follow-up
- Bullet points

Company: {company_name}

Latest Announcement Title:
{latest_title}

Latest Announcement Text:
{latest_clipped}

Previous Announcement Title:
{previous_title}

Previous Announcement Text:
{previous_clipped}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)