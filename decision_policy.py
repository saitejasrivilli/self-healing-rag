"""
cognition/decision_policy.py
==============================
Aggregates signals from Verification + Reflection to make the
final decision: ACCEPT | RETRY | ESCALATE.

Decision matrix:
  confidence ≥ threshold AND verified     → ACCEPT
  confidence < threshold, attempts < max  → RETRY
  confidence < threshold, attempts = max  → ESCALATE (return best with warning)
  reflection.needs_revision               → RETRY (if attempts remain)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Decision(str, Enum):
    ACCEPT   = "accept"
    RETRY    = "retry"
    ESCALATE = "escalate"


@dataclass
class PolicyInput:
    confidence: float
    verified: bool
    attempts: int
    max_retries: int
    confidence_threshold: float
    needs_revision: bool = False
    reflection_overall: float = 1.0


@dataclass
class PolicyOutput:
    decision: Decision
    reason: str
    should_retry: bool
    should_warn_user: bool


class DecisionPolicy:
    """
    Stateless policy evaluator.
    Takes signals from verifier + reflector → returns a Decision.
    """

    def evaluate(self, inp: PolicyInput) -> PolicyOutput:
        # Fast-path: high confidence + verified + no revision needed
        if (inp.confidence >= inp.confidence_threshold
                and inp.verified
                and not inp.needs_revision):
            return PolicyOutput(
                decision=Decision.ACCEPT,
                reason="Confidence and verification passed.",
                should_retry=False,
                should_warn_user=False,
            )

        # Can still retry
        if inp.attempts < inp.max_retries:
            reasons = []
            if inp.confidence < inp.confidence_threshold:
                reasons.append(f"confidence {inp.confidence:.2f} < threshold {inp.confidence_threshold:.2f}")
            if not inp.verified:
                reasons.append("answer not verified")
            if inp.needs_revision:
                reasons.append("reflection flagged revision needed")
            return PolicyOutput(
                decision=Decision.RETRY,
                reason="; ".join(reasons),
                should_retry=True,
                should_warn_user=False,
            )

        # Out of retries — escalate
        warn = inp.confidence < inp.confidence_threshold or not inp.verified
        return PolicyOutput(
            decision=Decision.ESCALATE,
            reason=f"Max retries ({inp.max_retries}) reached. Best confidence: {inp.confidence:.2f}.",
            should_retry=False,
            should_warn_user=warn,
        )

    def is_acceptable(self, confidence: float, threshold: float, verified: bool) -> bool:
        return confidence >= threshold and verified
