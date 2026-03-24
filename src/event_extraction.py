import json
import re
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import CHAT_MODEL, OLLAMA_BASE_URL


class ExtractedEvent(BaseModel):
    source_type: str = Field(description="Must be 'announcement' or 'report'")
    company_name: str = Field(description="Name of the company, e.g. John Keells")
    ticker: str = Field(description="The ticker symbol if known")
    event_type: str = Field(
        description="Must be one of: dividend, rights_issue, board_meeting, annual_report, interim_report, director_dealing, management_change, acquisition, disposal, capital_raising, profit_warning, litigation, governance, other"
    )
    event_subtype: str = Field(description="Specific sub-category or secondary detail")
    announcement_or_report_title: str = Field(description="The official title or headline")
    event_date: str = Field(description="Date the event was announced or occurred")
    effective_date: str = Field(description="Date the event takes effect, if applicable")
    materiality_level: str = Field(description="High, Medium, or Low")
    positive_signals: List[str] = Field(description="List of positive developments or signals")
    risk_signals: List[str] = Field(description="List of risks, negative developments, or bearish signals")
    key_numbers: List[str] = Field(description="List of critical quantitative metrics or financial figures mentioned")
    summary: str = Field(description="1-2 sentence core summary of the event")
    confidence: str = Field(description="Confidence in extraction: High, Medium, Low")
    unknowns: List[str] = Field(description="Details that are explicitly omitted or unclear from the text")


EVENT_PROMPT_TEMPLATE = """You are a financial data extraction engine for the Colombo Stock Exchange (CSE).
Given the following raw text from a public disclosure or report, extract the key structured event data.

Extract exactly matching the JSON schema provided below. Do not add any conversational text outside the JSON.

Text Metadata:
Company: {company_name}
Ticker: {ticker}
Source Title: {title}
Secondary Info: {secondary_info}

Raw Text:
{text}

JSON Formatting Instructions:
{format_instructions}
"""


def _get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        format="json", # Hint Ollama to return JSON if supported
    )


def normalize_event_output(raw_output: Any, fallback_type: str = "other") -> dict:
    """Robustly parse and normalize JSON output, falling back to a safe dict if malformed."""
    if isinstance(raw_output, dict):
        return raw_output
        
    text = str(raw_output).strip()
    
    # Strip markdown logic
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    try:
        return json.loads(text)
    except Exception:
        # Fallback skeleton
        return {
            "source_type": "unknown",
            "company_name": "Unknown",
            "ticker": "Unknown",
            "event_type": fallback_type,
            "event_subtype": "Parse Error",
            "announcement_or_report_title": "Extraction Failed",
            "event_date": "",
            "effective_date": "",
            "materiality_level": "Low",
            "positive_signals": [],
            "risk_signals": [],
            "key_numbers": [],
            "summary": "Failed to parse structured JSON. Raw text: " + text[:200],
            "confidence": "Low",
            "unknowns": ["Malformed extraction"],
        }


def _run_extraction_chain(company_name: str, ticker: str, title: str, secondary_info: str, text: str) -> dict:
    parser = JsonOutputParser(pydantic_object=ExtractedEvent)
    
    prompt = PromptTemplate(
        template=EVENT_PROMPT_TEMPLATE,
        input_variables=["company_name", "ticker", "title", "secondary_info", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    llm = _get_llm()
    chain = prompt | llm | parser
    
    try:
        raw_res = chain.invoke({
            "company_name": company_name or "Unknown",
            "ticker": ticker or "Unknown",
            "title": title or "Unknown",
            "secondary_info": secondary_info or "None",
            "text": text[:10000], # Restrict context window trivially
        })
        return normalize_event_output(raw_res)
    except Exception as e:
        return normalize_event_output(str(e))


def extract_events_from_announcement(company_name: str, ticker: str, title: str, category: str, text: str) -> dict:
    """Extract structured data specifically geared for announcements."""
    event = _run_extraction_chain(company_name, ticker, title, f"Category: {category}", text)
    event["source_type"] = "announcement"
    return event


def extract_events_from_report(company_name: str, ticker: str, report_type: str, text: str) -> dict:
    """Extract structured data specifically geared for full reports."""
    event = _run_extraction_chain(company_name, ticker, f"{report_type} Intelligence", f"Report Type: {report_type}", text)
    event["source_type"] = "report"
    return event


def event_importance_score(event: dict) -> str:
    """Heuristic logic to compute a consolidated Importance Score (High, Medium, Low)."""
    base_materiality = str(event.get("materiality_level", "Low")).strip().title()
    event_type = str(event.get("event_type", "other")).strip().lower()
    
    high_impact_types = {
        "dividend", "rights_issue", "profit_warning", "acquisition", "disposal", "capital_raising", "litigation"
    }
    
    if event_type in high_impact_types or base_materiality == "High":
        return "High"
        
    medium_impact_types = {"management_change", "director_dealing", "interim_report", "annual_report"}
    
    pos_count = len(event.get("positive_signals", []))
    risk_count = len(event.get("risk_signals", []))
    
    if event_type in medium_impact_types or base_materiality == "Medium" or pos_count > 2 or risk_count > 2:
        return "Medium"
        
    return "Low"


def event_to_markdown(event: dict) -> str:
    """Safely format an extracted event dictionary into Markdown format for LLM context or UI."""
    lines = [
        f"**Event Output**: {event.get('title', event.get('announcement_or_report_title', 'Unknown'))}",
        f"- **Type**: {event.get('event_type')} ({event.get('event_subtype')})",
        f"- **Company/Ticker**: {event.get('company_name')} / {event.get('ticker')}",
        f"- **Materiality**: {event.get('materiality_level')} (Computed Importance: {event_importance_score(event)})",
        f"- **Date Info**: {event.get('event_date')} (Effective: {event.get('effective_date')})",
        f"- **Summary**: {event.get('summary')}",
        "",
        "**Quantitative & Qualitative Signals**:",
    ]
    
    pos = event.get("positive_signals", [])
    if pos:
        lines.append("- *Positive*: " + "; ".join(str(p) for p in pos))
    
    risk = event.get("risk_signals", [])
    if risk:
        lines.append("- *Risks*: " + "; ".join(str(r) for r in risk))
        
    nums = event.get("key_numbers", [])
    if nums:
        lines.append("- *Metrics*: " + "; ".join(str(n) for n in nums))
        
    unks = event.get("unknowns", [])
    if unks:
        lines.append("- *Unknowns*: " + "; ".join(str(u) for u in unks))
        
    return "\n".join(lines)
