"""Pure-Python guardrail validator for drafted customer replies.

Ensures replies adhere to strict corporate compliance:
1. No promises of refunds, financial compensation, or free vouchers.
2. No admissions of legal liability or negligence.
3. No personal data / PII (emails, phone numbers).
4. Strict length bounds (minimum 20 characters, maximum 750 characters).
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List


@dataclass
class ValidationResult:
    """Outcome of validating a drafted reply."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    reply_text: str = ""

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "reply_text": self.reply_text,
        }


# Banned phrases and regexes for financial promises and legal admissions
BANNED_FINANCIAL_PATTERNS = [
    r"\b(?:promise|give|issue|send|provide|grant)\b.*?\b(?:full\s+refund|refund|reimbursement|money\s+back)\b",
    r"\b(?:100%\s+refund|total\s+refund|cash\s+refund)\b",
    r"\b(?:monetary\s+compensation|financial\s+compensation|financial\s+settlement)\b",
    r"\b(?:wire\s+\$?\d+|send\s+\$?\d+|pay\s+you\s+\$?\d+)\b",
    r"\b(?:free\s+meals?\s+for\s+life|free\s+service\s+forever|free\s+voucher)\b",
    r"\b(?:automatic\s+monetary\s+compensation)\b",
    r"\b(?:compensation\s+voucher)\b",
]

BANNED_LEGAL_PATTERNS = [
    r"\b(?:admit|admitting|accept)\b.*?\b(?:full\s+legal\s+liability|legal\s+liability|gross\s+negligence|guilt)\b",
    r"\b(?:we\s+were\s+grossly\s+negligent|our\s+gross\s+negligence)\b",
    r"\b(?:company\s+admits\s+(?:gross\s+negligence|legal\s+liability))\b",
    r"\b(?:we\s+are\s+legally\s+liable|we\s+admit\s+fault)\b",
    r"\b(?:settlement\s+offer|settle\s+out\s+of\s+court)\b",
]

# PII patterns
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)


class ReplyValidator:
    """Validates AI-generated or template customer replies."""

    def __init__(self, min_len: int = 20, max_len: int = 750):
        self.min_len = min_len
        self.max_len = max_len
        self.financial_regexes = [
            re.compile(p, re.IGNORECASE) for p in BANNED_FINANCIAL_PATTERNS
        ]
        self.legal_regexes = [
            re.compile(p, re.IGNORECASE) for p in BANNED_LEGAL_PATTERNS
        ]

    def validate(self, reply: str) -> ValidationResult:
        """Inspect reply for compliance violations."""
        if not reply or not isinstance(reply, str):
            return ValidationResult(
                passed=False,
                violations=["Empty or invalid reply string"],
                reply_text=reply or "",
            )

        trimmed = reply.strip()
        violations: List[str] = []

        # Length validation
        if len(trimmed) < self.min_len:
            violations.append(
                f"Reply is too short ({len(trimmed)} chars, minimum {self.min_len})"
            )
        if len(trimmed) > self.max_len:
            violations.append(
                f"Reply is too long ({len(trimmed)} chars, maximum {self.max_len})"
            )

        # Financial promise validation
        for rgx in self.financial_regexes:
            match = rgx.search(trimmed)
            if match:
                violations.append(
                    f"Banned financial compensation promise detected: '{match.group(0)}'"
                )
                break

        # Legal liability admission validation
        for rgx in self.legal_regexes:
            match = rgx.search(trimmed)
            if match:
                violations.append(
                    f"Banned legal liability admission detected: '{match.group(0)}'"
                )
                break

        # PII checks
        email_matches = EMAIL_PATTERN.findall(trimmed)
        if email_matches:
            violations.append(
                f"Personal email address detected in reply: {email_matches[0]}"
            )

        phone_matches = PHONE_PATTERN.findall(trimmed)
        if phone_matches:
            violations.append(
                f"Phone number detected in reply: {phone_matches[0]}"
            )

        return ValidationResult(
            passed=len(violations) == 0,
            violations=violations,
            reply_text=trimmed,
        )


_GLOBAL_VALIDATOR = ReplyValidator()


def get_reply_validator() -> ReplyValidator:
    """Return default ReplyValidator instance."""
    return _GLOBAL_VALIDATOR
