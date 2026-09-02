from __future__ import annotations

import re

# ponytail: regex redaction is enough for public-source POC data; switch to Presidio if audits
# show names, addresses, or other PHI patterns escaping these detectors.
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("phone", re.compile(r"(?:\+33|0)[1-9](?:[\s.-]?\d{2}){4}\b")),
    ("ssn_fr", re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b")),
    (
        "date_birth_hint",
        re.compile(r"\b(?:né|née|born)\s+(?:le\s+)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    ),
)


def find_pii(text: str) -> list[str]:
    """Return names of regex PII detectors that match the text."""

    return [name for name, pattern in PII_PATTERNS if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Replace detected PII spans with typed redaction markers."""

    redacted = text
    for name, pattern in PII_PATTERNS:
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted


def assert_no_pii(text: str) -> None:
    """Raise when text matches any configured PII detector."""

    matches = find_pii(text)
    if matches:
        raise ValueError(f"possible PII detected: {', '.join(matches)}")
