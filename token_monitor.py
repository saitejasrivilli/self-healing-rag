"""
safety/token_monitor.py
========================
Monitors token usage across all LLM calls to prevent
runaway costs and enforce rate limits.

Features:
  - Per-request token counting (prompt + completion)
  - Per-minute rolling window rate limiting
  - Cost estimation by model
  - Alert threshold callbacks
"""

from __future__ import annotations
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Approximate cost per 1M tokens (input / output) in USD
_COST_TABLE: dict[str, dict[str, float]] = {
    "gemini-1.5-flash":            {"input": 0.075,  "output": 0.30},
    "gemini-1.5-pro":              {"input": 1.25,   "output": 5.00},
    "gpt-4o":                      {"input": 2.50,   "output": 10.00},
    "gpt-4o-mini":                 {"input": 0.15,   "output": 0.60},
    "microsoft/Phi-3-mini-4k-instruct": {"input": 0.0, "output": 0.0},  # local
}


@dataclass
class TokenUsageRecord:
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    trace_id: str = ""


class TokenMonitor:
    """
    Thread-safe token usage tracker with rolling rate limits.
    """

    def __init__(
        self,
        alert_per_request: int = 4000,
        alert_per_minute: int = 50_000,
        alert_callback: Optional[Callable[[str, int], None]] = None,
    ):
        self.alert_per_request = alert_per_request
        self.alert_per_minute  = alert_per_minute
        self.alert_callback    = alert_callback or self._default_alert
        self._lock = threading.Lock()
        self._records: list[TokenUsageRecord] = []
        self._window: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        trace_id: str = "",
    ) -> TokenUsageRecord:
        total = prompt_tokens + completion_tokens
        cost  = self._estimate_cost(model, prompt_tokens, completion_tokens)

        record = TokenUsageRecord(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
            trace_id=trace_id,
        )

        with self._lock:
            self._records.append(record)
            self._window.append((record.timestamp, total))
            self._evict_old()

        # Alert checks
        if total > self.alert_per_request:
            self.alert_callback("per_request", total)
        per_min = self._tokens_last_minute()
        if per_min > self.alert_per_minute:
            self.alert_callback("per_minute", per_min)

        return record

    def _estimate_cost(self, model: str, prompt_t: int, completion_t: int) -> float:
        costs = _COST_TABLE.get(model, {"input": 0.0, "output": 0.0})
        return round(
            (prompt_t / 1_000_000) * costs["input"] +
            (completion_t / 1_000_000) * costs["output"],
            6,
        )

    def _tokens_last_minute(self) -> int:
        cutoff = time.time() - 60
        return sum(t for ts, t in self._window if ts >= cutoff)

    def _evict_old(self) -> None:
        cutoff = time.time() - 60
        while self._window and self._window[0][0] < cutoff:
            self._window.popleft()

    def summary(self) -> dict:
        with self._lock:
            if not self._records:
                return {"total_requests": 0}
            total_tokens = sum(r.total_tokens for r in self._records)
            total_cost   = sum(r.estimated_cost_usd for r in self._records)
            return {
                "total_requests":       len(self._records),
                "total_tokens":         total_tokens,
                "total_cost_usd":       round(total_cost, 4),
                "avg_tokens_per_req":   round(total_tokens / len(self._records), 1),
                "tokens_last_minute":   self._tokens_last_minute(),
            }

    @staticmethod
    def _default_alert(alert_type: str, count: int) -> None:
        logger.warning("TokenMonitor ALERT [%s]: %d tokens", alert_type, count)


# Module-level singleton
token_monitor = TokenMonitor()
