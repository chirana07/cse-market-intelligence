from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse


MAX_URL_LENGTH = 2048

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\breveal\s+(the\s+)?(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(the\s+)?(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(in|acting\s+as)\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b|\bsecret\s+key\b|\bapi\s+key\b", re.IGNORECASE),
    re.compile(r"<\s*script\b|javascript\s*:", re.IGNORECASE),
]

HIGH_RISK_PATTERNS = [
    re.compile(r"\breveal\s+(the\s+)?(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bprint\s+(the\s+)?(system|developer)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bexfiltrate\b|\bsecret\s+key\b|\bapi\s+key\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptInjectionAssessment:
    risk_level: str
    matched_patterns: list[str] = field(default_factory=list)
    should_block: bool = False


def _is_private_or_local_hostname(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True

    if host in {"localhost", "0.0.0.0"} or host.endswith(".localhost"):
        return True

    try:
        ip = ipaddress.ip_address(host)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
    except ValueError:
        return False


def validate_ingest_url(url: str) -> GuardrailDecision:
    candidate = (url or "").strip()
    if not candidate:
        return GuardrailDecision(False, "URL is empty.")
    if len(candidate) > MAX_URL_LENGTH:
        return GuardrailDecision(False, "URL is too long.")

    try:
        parsed = urlparse(candidate)
    except Exception:
        return GuardrailDecision(False, "URL could not be parsed.")

    if parsed.scheme not in {"http", "https"}:
        return GuardrailDecision(False, "Only http and https URLs are allowed.")
    if not parsed.netloc or not parsed.hostname:
        return GuardrailDecision(False, "URL must include a hostname.")
    if parsed.username or parsed.password:
        return GuardrailDecision(False, "URLs with embedded credentials are not allowed.")
    if _is_private_or_local_hostname(parsed.hostname):
        return GuardrailDecision(False, "Private, local, or reserved network targets are not allowed.")

    return GuardrailDecision(True)


def assess_prompt_injection(text: str, *, block_high_risk: bool = True) -> PromptInjectionAssessment:
    value = text or ""
    matched = []
    high_risk = False

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(value):
            matched.append(pattern.pattern)

    for pattern in HIGH_RISK_PATTERNS:
        if pattern.search(value):
            high_risk = True
            break

    if high_risk:
        return PromptInjectionAssessment(
            risk_level="high",
            matched_patterns=matched,
            should_block=block_high_risk,
        )
    if matched:
        return PromptInjectionAssessment(
            risk_level="medium",
            matched_patterns=matched,
            should_block=False,
        )
    return PromptInjectionAssessment(risk_level="low")
