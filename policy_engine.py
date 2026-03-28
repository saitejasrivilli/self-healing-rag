"""
safety/policy_engine.py
========================
Central policy enforcement point for the platform.
Combines InputGuard + OutputGuard + rate limiting + access control.

All requests pass through PolicyEngine.check_input() before the pipeline
and PolicyEngine.check_output() before the response is returned.
"""

from __future__ import annotations
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from safety.input_guard import InputGuard, InputGuardResult
from safety.output_guard import OutputGuard, OutputGuardResult

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    allowed: bool
    reason: Optional[str]
    flags: list[str]


class PolicyEngine:
    def __init__(
        self,
        rate_limit_per_minute: int = 60,
        max_input_length: int = 2000,
        block_pii_input: bool = False,
        redact_pii_output: bool = True,
        block_toxic_output: bool = True,
        blocked_keywords: list[str] | None = None,
    ):
        self.input_guard = InputGuard(
            max_length=max_input_length,
            block_pii=block_pii_input,
            blocked_keywords=blocked_keywords,
        )
        self.output_guard = OutputGuard(
            redact_pii=redact_pii_output,
            block_toxic=block_toxic_output,
        )
        self._rate_limit = rate_limit_per_minute
        self._request_times: dict[str, list[float]] = defaultdict(list)

    # ── Input ─────────────────────────────────────────────────────────────────
    def check_input(self, text: str, client_id: str = "default") -> tuple[PolicyDecision, str]:
        """
        Returns (PolicyDecision, sanitized_text).
        Call before passing user input to the pipeline.
        """
        # Rate limit
        if not self._rate_ok(client_id):
            return (
                PolicyDecision(allowed=False, reason="Rate limit exceeded.", flags=["rate_limited"]),
                text,
            )

        result: InputGuardResult = self.input_guard.check(text)
        decision = PolicyDecision(
            allowed=result.allowed,
            reason=result.blocked_reason,
            flags=result.flags,
        )
        return decision, result.sanitized_input

    # ── Output ────────────────────────────────────────────────────────────────
    def check_output(self, answer: str, confidence: float = 1.0) -> tuple[PolicyDecision, str]:
        """
        Returns (PolicyDecision, cleaned_answer).
        Call before returning the LLM answer to the user.
        """
        result: OutputGuardResult = self.output_guard.check(answer, confidence)
        decision = PolicyDecision(
            allowed=result.allowed,
            reason=result.blocked_reason,
            flags=result.flags,
        )
        return decision, result.answer

    # ── Rate limiting ─────────────────────────────────────────────────────────
    def _rate_ok(self, client_id: str) -> bool:
        now = time.time()
        window = self._request_times[client_id]
        window[:] = [t for t in window if now - t < 60]
        if len(window) >= self._rate_limit:
            logger.warning("Rate limit exceeded for client '%s'", client_id)
            return False
        window.append(now)
        return True

    def stats(self) -> dict:
        return {
            "active_clients": len(self._request_times),
            "rate_limit_per_minute": self._rate_limit,
        }


# Module-level singleton (configure once at startup)
policy = PolicyEngine()
