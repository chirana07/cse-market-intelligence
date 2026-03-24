import json
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.config import CHAT_MODEL, OLLAMA_BASE_URL

# Schema version — increment when field definitions change
FINANCIAL_SCHEMA_VERSION = "v1"


class ExtractedFinancials(BaseModel):
    source_type: str = Field(description="Must be 'announcement' or 'report'")
    company_name: str = Field(description="Name of the company")
    ticker: str = Field(description="Ticker symbol if known")
    document_title: str = Field(description="Title of the document")
    document_date: str = Field(description="Date of the document or event")
    reporting_period: str = Field(default="Unknown", description="E.g., Q1 2024, FY23")
    currency: str = Field(default="LKR", description="Currency of metrics")
    
    revenue: Optional[str] = Field(default=None, description="Total revenue or top-line income")
    revenue_growth_pct: Optional[str] = Field(default=None, description="Percentage growth in revenue")
    gross_profit: Optional[str] = Field(default=None)
    operating_profit: Optional[str] = Field(default=None)
    net_profit: Optional[str] = Field(default=None)
    net_profit_growth_pct: Optional[str] = Field(default=None)
    eps: Optional[str] = Field(default=None, description="Earnings per share")
    eps_growth_pct: Optional[str] = Field(default=None)
    total_assets: Optional[str] = Field(default=None)
    total_liabilities: Optional[str] = Field(default=None)
    equity: Optional[str] = Field(default=None)
    operating_cash_flow: Optional[str] = Field(default=None)
    free_cash_flow: Optional[str] = Field(default=None)
    dividend_per_share: Optional[str] = Field(default=None)
    
    payout_signal: Optional[str] = Field(default=None, description="Signal on dividends or share buybacks")
    guidance_signal: Optional[str] = Field(default=None, description="Future financial guidance or outlook")
    margin_signal: Optional[str] = Field(default=None, description="Trends in gross or operating margins")
    leverage_signal: Optional[str] = Field(default=None, description="Debt or leverage shifts")
    liquidity_signal: Optional[str] = Field(default=None, description="Cash flow or liquidity stance")
    
    management_tone: str = Field(default="Neutral", description="Optimistic, Neutral, or Cautious")
    positive_signals: List[str] = Field(default_factory=list, description="Specific positive financial achievements")
    risk_signals: List[str] = Field(default_factory=list, description="Specific financial risks or headwinds")
    strategic_signals: List[str] = Field(default_factory=list, description="Major strategic financial moves")
    key_numbers: List[str] = Field(default_factory=list, description="Other critical unmapped numbers")
    notes: str = Field(default="")
    
    confidence: str = Field(default="High", description="High, Medium, or Low")
    unknowns: List[str] = Field(default_factory=list, description="Which highly relevant metrics were missing")


FINANCIAL_PROMPT_TEMPLATE = """You are a specialized equity research data extraction engine.
Given the following raw text from a public disclosure or report, extract the pure financial metrics into a structured format.

Constraint: If a numeric value is missing, return null or explicitly state "Unknown". DO NOT hallucinate numbers.

Text Metadata:
Company: {company_name}
Ticker: {ticker}
Source Title: {title}
Secondary Context: {context}

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
        format="json",  # Hint to Ollama to strictly return JSON
    )


def normalize_financial_fact_output(raw_output: Any, source_type: str = "unknown") -> dict:
    if isinstance(raw_output, dict):
        return raw_output
        
    text = str(raw_output).strip()
    
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    try:
        data = json.loads(text)
        if not data.get("source_type"):
            data["source_type"] = source_type
        return data
    except Exception:
        return {
            "source_type": source_type,
            "company_name": "Unknown",
            "ticker": "Unknown",
            "document_title": "Extraction Failed",
            "document_date": "",
            "reporting_period": "Unknown",
            "currency": "",
            "management_tone": "Unknown",
            "confidence": "Low",
            "unknowns": ["Malformed extraction output - JSON block failed parse."],
            "notes": "Failed to parse structured JSON. Raw text snippet: " + text[:200]
        }


def _run_financial_extraction(company_name: str, ticker: str, title: str, context: str, text: str) -> dict:
    parser = JsonOutputParser(pydantic_object=ExtractedFinancials)
    
    prompt = PromptTemplate(
        template=FINANCIAL_PROMPT_TEMPLATE,
        input_variables=["company_name", "ticker", "title", "context", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    llm = _get_llm()
    chain = prompt | llm | parser
    
    try:
        raw_res = chain.invoke({
            "company_name": company_name or "Unknown",
            "ticker": ticker or "Unknown",
            "title": title or "Unknown",
            "context": context or "None",
            "text": text[:12000],  # Bound context safely
        })
        return normalize_financial_fact_output(raw_res)
    except Exception as e:
        return normalize_financial_fact_output(str(e))


def extract_financial_facts_from_announcement(company_name: str, ticker: str, title: str, category: str, text: str) -> dict:
    fin = _run_financial_extraction(company_name, ticker, title, f"Category: {category}", text)
    fin["source_type"] = "announcement"
    return fin


def extract_financial_facts_from_report(company_name: str, ticker: str, report_type: str, text: str) -> dict:
    fin = _run_financial_extraction(company_name, ticker, f"{report_type} Financials", f"Report: {report_type}", text)
    fin["source_type"] = "report"
    return fin


def financial_fact_to_markdown(fin: dict) -> str:
    """Formatter to safely convert a parsed financial dict into text for subsequent AI context ingestion."""
    lines = [
        f"**Financial Extraction**: {fin.get('document_title', 'Unknown Title')} ({fin.get('company_name')} - {fin.get('ticker')})",
        f"- **Period**: {fin.get('reporting_period')} | **Currency**: {fin.get('currency')} | **Tone**: {fin.get('management_tone')}",
        "",
        "**Key Metrics**:"
    ]
    
    metrics = {
        "Revenue": fin.get("revenue"),
        "Rev Growth %": fin.get("revenue_growth_pct"),
        "Gross Profit": fin.get("gross_profit"),
        "Operating Profit": fin.get("operating_profit"),
        "Net Profit": fin.get("net_profit"),
        "Net Profit %": fin.get("net_profit_growth_pct"),
        "EPS": fin.get("eps"),
        "Dividend/Share": fin.get("dividend_per_share")
    }
    
    for k, v in metrics.items():
        if v and v != "Unknown":
            lines.append(f"- {k}: {v}")
            
    lines.append("")
    lines.append("**Qualitative Signals**:")
    
    for sig_name, sig_key in [("Payout", "payout_signal"), ("Margin", "margin_signal"), ("Guidance", "guidance_signal"), ("Liquidity", "liquidity_signal")]:
        v = fin.get(sig_key)
        if v and v != "Unknown":
            lines.append(f"- *{sig_name}*: {v}")
            
    pos = fin.get("positive_signals", [])
    if isinstance(pos, list) and pos:
        lines.append("- *Positives*: " + "; ".join(str(p) for p in pos))
        
    risk = fin.get("risk_signals", [])
    if isinstance(risk, list) and risk:
        lines.append("- *Risks*: " + "; ".join(str(r) for r in risk))
        
    return "\n".join(lines)


def merge_financial_fact_objects(base: dict, overlay: dict) -> dict:
    """Merge two financial fact dicts, preferring non-null values from `overlay` over `base`.
    
    Use this to combine a cached report extraction with a fresh announcement extraction
    so the most recent data wins for each field without dropping older context.
    """
    if not base:
        return overlay or {}
    if not overlay:
        return base

    merged = dict(base)  # Start from base
    
    # Scalar financial fields — overlay wins if non-null and non-unknown
    scalar_fields = [
        "revenue", "revenue_growth_pct", "gross_profit", "operating_profit",
        "net_profit", "net_profit_growth_pct", "eps", "eps_growth_pct",
        "total_assets", "total_liabilities", "equity", "operating_cash_flow",
        "free_cash_flow", "dividend_per_share",
        "payout_signal", "guidance_signal", "margin_signal", "leverage_signal", "liquidity_signal",
        "management_tone", "reporting_period", "currency", "confidence",
    ]
    for field in scalar_fields:
        overlay_val = overlay.get(field)
        if overlay_val and overlay_val not in ("Unknown", "unknown", None, ""):
            merged[field] = overlay_val

    # List fields — union, dedup, preserve order
    list_fields = ["positive_signals", "risk_signals", "strategic_signals", "key_numbers", "unknowns"]
    for field in list_fields:
        base_list = base.get(field) or []
        overlay_list = overlay.get(field) or []
        if isinstance(base_list, list) and isinstance(overlay_list, list):
            seen = set()
            combined = []
            for item in base_list + overlay_list:
                s = str(item)
                if s not in seen:
                    seen.add(s)
                    combined.append(item)
            merged[field] = combined

    merged["schema_version"] = FINANCIAL_SCHEMA_VERSION
    return merged
