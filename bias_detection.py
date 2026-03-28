"""
safety/bias_detection.py
=========================
Lightweight bias detection for LLM inputs and outputs.

Detects:
  - Demographic bias signals (gendered assumptions, racial stereotypes)
  - Political bias (one-sided framing)
  - Sentiment extremes (overly positive/negative)

This is a keyword-heuristic baseline. For production,
swap the _score methods with a fine-tuned classifier
(e.g. Detoxify, Perspective API, or a HF classifier).
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Gendered bias signals (oversimplifications)
_GENDER_BIAS = [
    r"\b(all women|all men|women are|men are|females are|males are)\b",
    r"\b(typical (woman|man|female|male))\b",
]

# Racial / ethnic stereotyping signals
_RACE_BIAS = [
    r"\b(all (black|white|asian|hispanic|latino) (people|men|women))\b",
    r"\b(those people)\b",
]

# Political framing signals
_POLITICAL_BIAS = [
    r"\b(radical (left|right)|extreme (liberal|conservative))\b",
    r"\b((democrats|republicans) always|always (democrats|republicans))\b",
]

# Sentiment extremes
_EXTREME_POSITIVE = ["always", "best ever", "never fails", "perfect", "flawless"]
_EXTREME_NEGATIVE = ["always fails", "terrible", "worst ever", "absolutely horrible", "disgusting"]


@dataclass
class BiasReport:
    has_bias: bool
    bias_types: list[str]
    severity: float              # 0.0 (none) – 1.0 (high)
    details: list[str]
    recommendation: Optional[str] = None


class BiasDetector:
    def __init__(self, sensitivity: float = 0.5):
        """
        sensitivity: 0.0 = lenient, 1.0 = very strict.
        """
        self.sensitivity = sensitivity
        self._patterns = {
            "gender_bias":    [re.compile(p, re.IGNORECASE) for p in _GENDER_BIAS],
            "racial_bias":    [re.compile(p, re.IGNORECASE) for p in _RACE_BIAS],
            "political_bias": [re.compile(p, re.IGNORECASE) for p in _POLITICAL_BIAS],
        }

    def analyze(self, text: str) -> BiasReport:
        bias_types: list[str] = []
        details: list[str] = []
        hits = 0

        for bias_type, patterns in self._patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    bias_types.append(bias_type)
                    details.append(f"{bias_type}: matched '{match.group(0)}'")
                    hits += 1
                    break   # one flag per type

        # Sentiment extremes
        lower = text.lower()
        pos_hits = sum(1 for p in _EXTREME_POSITIVE if p in lower)
        neg_hits = sum(1 for p in _EXTREME_NEGATIVE if p in lower)
        if pos_hits >= 3 or neg_hits >= 2:
            bias_types.append("sentiment_extreme")
            details.append(f"sentiment_extreme: {pos_hits} pos / {neg_hits} neg signals")
            hits += 1

        severity = min(hits * 0.3 * (1 + self.sensitivity), 1.0)
        has_bias = severity > (1 - self.sensitivity) * 0.5

        recommendation = None
        if has_bias:
            recommendation = (
                "Consider rephrasing to avoid generalizations. "
                "Use specific, evidence-based language."
            )

        return BiasReport(
            has_bias=has_bias,
            bias_types=list(set(bias_types)),
            severity=round(severity, 3),
            details=details,
            recommendation=recommendation,
        )


# Module-level singleton
bias_detector = BiasDetector()
