"""
safety/output_guard.py
=======================
Validates LLM-generated answers before returning them to users.

Checks:
  - Hallucination signals ("I think", "probably", "I'm not sure")
  - PII leakage in output
  - Toxic language (keyword-based)
  - Answer length sanity
  - Confidence threshold enforcement
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_UNCERTAINTY_PHRASES = [
    "i think", "i believe", "i'm not sure", "i'm not certain",
    "probably", "maybe", "might be", "could be", "i don't know",
    "i cannot determine", "not mentioned", "no information",
]

_TOXIC_PATTERNS = [
    r"\b(hate|kill|murder|attack|harm|abuse|exploit)\b",
]

_PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "ssn":   r"\b\d{3}-\d{2}-\d{4}\b",
}


@dataclass
class OutputGuardResult:
    allowed: bool
    answer: str
    flags: list[str]
    uncertainty_score: float    # 0.0–1.0 (higher = more uncertain language)
    blocked_reason: Optional[str] = None


class OutputGuard:
    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 3000,
        block_toxic: bool = True,
        redact_pii: bool = True,
        confidence_threshold: float = 0.0,   # 0 = don't enforce
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.block_toxic = block_toxic
        self.redact_pii = redact_pii
        self.confidence_threshold = confidence_threshold
        self._toxic_re = [re.compile(p, re.IGNORECASE) for p in _TOXIC_PATTERNS]
        self._pii_re = {k: re.compile(v) for k, v in _PII_PATTERNS.items()}

    def check(self, answer: str, confidence: float = 1.0) -> OutputGuardResult:
        flags: list[str] = []
        text = answer.strip()

        # Length checks
        if len(text) < self.min_length:
            return OutputGuardResult(allowed=False, answer=text, flags=["too_short"],
                                     uncertainty_score=0.0,
                                     blocked_reason="Answer too short.")
        if len(text) > self.max_length:
            text = text[:self.max_length] + "..."
            flags.append("truncated")

        # Confidence gate
        if self.confidence_threshold > 0 and confidence < self.confidence_threshold:
            flags.append("low_confidence")

        # Uncertainty score
        lower = text.lower()
        hits = sum(1 for phrase in _UNCERTAINTY_PHRASES if phrase in lower)
        uncertainty_score = min(hits / max(len(_UNCERTAINTY_PHRASES), 1) * 3, 1.0)
        if uncertainty_score > 0.3:
            flags.append("high_uncertainty")

        # Toxic language
        if self.block_toxic:
            for pattern in self._toxic_re:
                if pattern.search(text):
                    return OutputGuardResult(
                        allowed=False, answer=text,
                        flags=flags + ["toxic_content"],
                        uncertainty_score=uncertainty_score,
                        blocked_reason="Answer contains potentially harmful content.",
                    )

        # PII in output — redact
        if self.redact_pii:
            for pii_type, pattern in self._pii_re.items():
                if pattern.search(text):
                    text = pattern.sub(f"[REDACTED_{pii_type.upper()}]", text)
                    flags.append(f"pii_redacted_{pii_type}")

        return OutputGuardResult(
            allowed=True, answer=text,
            flags=flags, uncertainty_score=round(uncertainty_score, 3),
        )
