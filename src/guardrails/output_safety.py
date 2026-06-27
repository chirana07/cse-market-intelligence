from __future__ import annotations

import re
from dataclasses import dataclass, field


REQUIRED_ANALYST_SECTIONS = [
    "1. Direct Answer",
    "2. Why It Matters",
    "3. Key Evidence",
    "4. Risks / Unknowns",
    "5. Follow-up Questions",
]

DIRECT_ADVICE_PATTERNS = [
    re.compile(r"\b(you\s+should|must|need\s+to)\s+(buy|sell|short|dump|accumulate)\b", re.IGNORECASE),
    re.compile(r"\b(strong\s+buy|guaranteed\s+return|risk-free|cannot\s+lose)\b", re.IGNORECASE),
    re.compile(r"\bthis\s+stock\s+will\s+(definitely|certainly)\b", re.IGNORECASE),
]


@dataclass(frozen=True)
class AnswerValidation:
    structure_score: int
    missing_sections: list[str] = field(default_factory=list)
    unsafe_advice_patterns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missing_sections and not self.unsafe_advice_patterns

    @property
    def status(self) -> str:
        if self.unsafe_advice_patterns:
            return "caution"
        if self.missing_sections:
            return "format_warning"
        return "passed"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "passed": self.passed,
            "structure_score": self.structure_score,
            "missing_sections": self.missing_sections,
            "unsafe_advice_patterns": self.unsafe_advice_patterns,
            "warnings": self.warnings,
        }


def validate_analyst_answer(answer: str) -> AnswerValidation:
    text = answer or ""
    lower_text = text.lower()

    found_sections = [
        section for section in REQUIRED_ANALYST_SECTIONS
        if section.lower() in lower_text
    ]
    missing_sections = [
        section for section in REQUIRED_ANALYST_SECTIONS
        if section not in found_sections
    ]
    structure_score = int((len(found_sections) / len(REQUIRED_ANALYST_SECTIONS)) * 100)

    unsafe_patterns = [
        pattern.pattern for pattern in DIRECT_ADVICE_PATTERNS
        if pattern.search(text)
    ]

    warnings = []
    if missing_sections:
        warnings.append("Analyst answer is missing one or more required output sections.")
    if unsafe_patterns:
        warnings.append("Analyst answer contains direct investment advice or certainty language.")

    return AnswerValidation(
        structure_score=structure_score,
        missing_sections=missing_sections,
        unsafe_advice_patterns=unsafe_patterns,
        warnings=warnings,
    )


def apply_output_safety_notice(answer: str, validation: AnswerValidation) -> str:
    if not validation.unsafe_advice_patterns:
        return answer

    notice = (
        "Compliance note: this output was flagged for direct investment-advice language. "
        "Treat it as research support only, not a buy/sell instruction.\n\n"
    )
    if answer.startswith("Compliance note:"):
        return answer
    return notice + answer
