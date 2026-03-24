from datetime import datetime, timezone
from urllib.parse import urlparse
import hashlib
import re


EVENT_KEYWORDS = {
    "Earnings": [
        "earnings", "quarter", "quarterly", "annual results", "revenue",
        "profit", "net income", "ebitda", "eps", "results"
    ],
    "Guidance": [
        "guidance", "outlook", "forecast", "expects", "projection",
        "target", "targets", "visibility"
    ],
    "M&A": [
        "acquisition", "acquire", "merger", "buyout", "takeover",
        "stake purchase", "divestment", "sell unit", "strategic sale"
    ],
    "Capital Markets": [
        "ipo", "rights issue", "share issue", "bond", "debt", "equity raise",
        "fundraising", "capital raise", "listing"
    ],
    "Management": [
        "ceo", "cfo", "chairman", "chairperson", "board", "director",
        "appointment", "resigned", "resignation", "management"
    ],
    "Regulation": [
        "regulator", "regulatory", "approval", "license", "compliance",
        "tax", "court", "lawsuit", "legal", "government", "policy"
    ],
    "Macro": [
        "inflation", "interest rate", "fx", "exchange rate", "currency",
        "gdp", "economy", "macro", "oil prices", "commodity prices"
    ],
    "Operations": [
        "capacity", "plant", "factory", "production", "shipment",
        "supply chain", "operations", "margin", "utilization"
    ],
    "Partnership/Product": [
        "partnership", "joint venture", "launch", "rollout", "contract",
        "customer win", "deal", "agreement", "product"
    ],
    "Dividend/Shareholder Return": [
        "dividend", "buyback", "shareholder return", "payout"
    ],
}

TICKER_STOPWORDS = {
    "CEO", "CFO", "USD", "LKR", "EUR", "GDP", "IPO", "EBITDA", "EPS",
    "Q1", "Q2", "Q3", "Q4", "YOY", "MOM", "FY", "SEC", "CSE", "NSE"
}


def clean_title(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def slug_to_title(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "Untitled Article"

    slug = path.split("/")[-1]
    slug = slug.replace("-", " ").replace("_", " ")
    slug = re.sub(r"\s+", " ", slug).strip()

    return slug.title() if slug else "Untitled Article"


def extract_title(doc, url: str) -> str:
    candidates = [
        doc.metadata.get("title"),
        doc.metadata.get("page_title"),
        doc.metadata.get("og:title"),
    ]

    for candidate in candidates:
        candidate = clean_title(candidate or "")
        if candidate:
            return candidate

    content = (doc.page_content or "").strip()
    if content:
        first_line = clean_title(content.splitlines()[0].strip())
        if 10 <= len(first_line) <= 180:
            return first_line

    return slug_to_title(url)


def detect_event_tags(title: str, text: str) -> list[str]:
    haystack = f"{title}\n{text[:5000]}".lower()
    tags = []

    for tag, keywords in EVENT_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            tags.append(tag)

    return tags if tags else ["General Update"]


def extract_possible_tickers(title: str, text: str) -> list[str]:
    haystack = f"{title}\n{text[:5000]}"
    candidates = []

    patterns = [
        r"(?:ticker|symbol|stock code|listed as|trades as)\s*[:\-]?\s*([A-Z]{2,6})\b",
        r"\(([A-Z]{2,6})\)",
        r"\b([A-Z]{2,6})\.[A-Z]{1,4}\b",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, haystack):
            token = match[0] if isinstance(match, tuple) else match
            token = token.split(".")[0].upper().strip()

            if not token.isalpha():
                continue
            if not (2 <= len(token) <= 6):
                continue
            if token in TICKER_STOPWORDS:
                continue

            candidates.append(token)

    unique = sorted(set(candidates))
    return unique[:8]


def format_label_list(values: list[str]) -> str:
    return " | ".join(values) if values else ""


def enrich_document_metadata(doc, url: str):
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    title = extract_title(doc, url)
    text = doc.page_content or ""

    doc_id = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()
    tickers = extract_possible_tickers(title, text)
    event_tags = detect_event_tags(title, text)

    doc.metadata["source"] = url
    doc.metadata["url"] = url
    doc.metadata["domain"] = domain
    doc.metadata["title"] = title
    doc.metadata["doc_id"] = doc_id
    doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
    doc.metadata["source_type"] = "url"

    doc.metadata["ticker_candidates"] = tickers
    doc.metadata["ticker_candidates_str"] = format_label_list(tickers)
    doc.metadata["primary_ticker"] = tickers[0] if tickers else "Unknown"

    doc.metadata["event_tags"] = event_tags
    doc.metadata["event_tags_str"] = format_label_list(event_tags)
    doc.metadata["primary_event"] = event_tags[0] if event_tags else "General Update"