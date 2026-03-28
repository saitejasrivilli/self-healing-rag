"""
safety/input_guard.py
======================
Validates and sanitises user input before it enters the pipeline.

Checks:
  - Length limits
  - Prompt injection patterns
  - Blocked topic keywords
  - PII detection (email, phone, SSN patterns)
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Prompt injection patterns (non-exhaustive — extend as needed)
_INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"act as (a |an )?(?!ai|assistant)",
    r"forget (everything|all|your)",
    r"system prompt",
    r"jailbreak",
    r"bypass.*filter",
]

# PII patterns
_PII_PATTERNS = {
    "email":   r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":   r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":     r"\b\d{3}-\d{2}-\d{4}\b",
    "cc":      r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
}


@dataclass
class InputGuardResult:
    allowed: bool
    sanitized_input: str
    flags: list[str]
    blocked_reason: Optional[str] = None


class InputGuard:
    def __init__(
        self,
        max_length: int = 2000,
        block_pii: bool = False,
        blocked_keywords: list[str] | None = None,
    ):
        self.max_length = max_length
        self.block_pii = block_pii
        self.blocked_keywords = [k.lower() for k in (blocked_keywords or [])]
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
        self._pii_re = {k: re.compile(v) for k, v in _PII_PATTERNS.items()}

    def check(self, text: str) -> InputGuardResult:
        flags: list[str] = []
        sanitized = text

        # Length check
        if len(text) > self.max_length:
            sanitized = text[: self.max_length]
            flags.append(f"truncated_to_{self.max_length}")

        # Prompt injection
        for pattern in self._injection_re:
            if pattern.search(sanitized):
                logger.warning("InputGuard: injection pattern detected")
                return InputGuardResult(
                    allowed=False,
                    sanitized_input=sanitized,
                    flags=flags + ["prompt_injection"],
                    blocked_reason="Potential prompt injection detected.",
                )

        # Blocked keywords
        lower = sanitized.lower()
        for kw in self.blocked_keywords:
            if kw in lower:
                return InputGuardResult(
                    allowed=False,
                    sanitized_input=sanitized,
                    flags=flags + ["blocked_keyword"],
                    blocked_reason=f"Query contains blocked keyword: '{kw}'.",
                )

        # PII detection
        for pii_type, pattern in self._pii_re.items():
            if pattern.search(sanitized):
                flags.append(f"pii_{pii_type}")
                if self.block_pii:
                    return InputGuardResult(
                        allowed=False,
                        sanitized_input=sanitized,
                        flags=flags,
                        blocked_reason=f"Query contains {pii_type.upper()} — blocked by policy.",
                    )
                # Redact instead of block
                sanitized = pattern.sub(f"[REDACTED_{pii_type.upper()}]", sanitized)

        return InputGuardResult(allowed=True, sanitized_input=sanitized, flags=flags)
