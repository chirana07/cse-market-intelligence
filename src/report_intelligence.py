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


def extract_interim_key_figures(
    company_name: str,
    ticker: str,
    report_text: str,
) -> str:
    """Extract investor key figures, growth, margins, main drivers, and investor snapshot from an interim financial report."""
    llm = _get_llm()
    clipped = (report_text or "")[:25000]

    prompt = f"""
You are a financial analyst with strong knowledge of accounting principles, financial statement analysis, and interim financial reporting.

I will provide you with an interim financial report of a company. Your job is to extract only the most important financial figures and performance indicators that an investor would need to quickly understand the company's performance.

## 1. First identify the reporting periods correctly
Before calculating anything, determine:
* Current reporting quarter
* Previous quarter
* Corresponding quarter of the previous year
* Current year-to-date (YTD) period
* Corresponding YTD period of the previous year

Be extremely careful with statements containing both:
* "Quarter ended" figures, and
* "Period/Year-to-date ended" figures

Never compare a 3-month figure with a 6-month, 9-month, or 12-month cumulative figure.

## 2. Extract the most important figures
Focus primarily on:

### Revenue
* Revenue for the current quarter
* Revenue for the previous quarter, if available
* Revenue for the same quarter last year
* Quarter-on-Quarter (QoQ) revenue growth %: ((Current Quarter Revenue - Previous Quarter Revenue) / Previous Quarter Revenue) * 100
* Year-on-Year (YoY) revenue growth %: ((Current Quarter Revenue - Same Quarter Last Year Revenue) / Same Quarter Last Year Revenue) * 100

### Profit After Tax – PAT
Extract:
* Current quarter PAT, previous quarter PAT (if available), same quarter last year PAT
* QoQ PAT growth and YoY PAT growth
* If the company moves from a loss to a profit or from a profit to a loss, do NOT give a misleading percentage increase/decrease. Instead state:
  - Turned profitable
  - Turned loss-making
  - Loss narrowed
  - Loss widened
  as appropriate.

### Profitability
Extract when available:
* Gross Profit, Operating Profit, Profit Before Tax (PBT), Profit After Tax (PAT)
Calculate useful margins:
* Gross Margin = Gross Profit / Revenue * 100
* Operating Margin = Operating Profit / Revenue * 100
* Net Profit Margin = PAT / Revenue * 100
Mention significant margin expansion or contraction compared with the relevant previous period.

## 3. Other important financial figures
Only include these when they are available and materially useful:
* EPS, Total Assets, Total Liabilities, Shareholders' Equity / Net Assets, Cash and Cash Equivalents, Borrowings / Interest-bearing debt, Finance Costs, Operating Cash Flow, Capital Expenditure, Dividend per share.
For banks, finance companies, insurers, or other specialized businesses, adapt the analysis to relevant accounting measures.

## 4. Identify the main drivers
Read the income statement and notes to identify WHY profits changed (revenue growth or decline, cost of sales changes, gross margin changes, admin expenses, selling/distribution expenses, finance costs, other income, foreign exchange gains/losses, fair-value gains/losses, impairment charges, tax changes, one-off/non-recurring gains or expenses).
Separate operating improvement from profits caused mainly by one-off items.

## 5. Accounting rules
Do NOT:
* Mix quarterly figures with YTD figures.
* Treat revenue as profit, PBT as PAT, OCI as PAT, cash flow as accounting profit.
* Calculate misleading growth percentages when the denominator is negative.
* Assume missing figures or invent numbers.
* Double-count subsidiaries or group/company figures.
Use Consolidated / Group figures unless specifically told otherwise.
If figures are unclear, state that clearly.

## 6. Output format
Keep the answer concise and investor-focused.

### [COMPANY NAME] – [Reporting Quarter]

| Metric | Current Quarter | Previous Quarter | QoQ | Same Quarter Last Year | YoY |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Revenue | | | | | |
| Gross Profit | | | | | |
| Operating Profit | | | | | |
| PBT | | | | | |
| PAT | | | | | |
| EPS | | | | | |

Use N/A where a valid comparison cannot be established from the report.

Then provide:

### Key Takeaways
Give only 3–6 important observations (e.g., Revenue, PAT, Net margin, Finance costs, Cash/borrowings, Main driver).

### Investor Snapshot
Finish with:
**Overall performance:** Strong / Improving / Stable / Weakening / Weak
**Reason:** Explain the classification in 1–2 sentences based strictly on the financial statements.
Do NOT give a Buy/Sell/Hold recommendation.

Company Name: {company_name or "Not specified"}
Ticker: {ticker or "Not specified"}

Report Text:
{clipped}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)