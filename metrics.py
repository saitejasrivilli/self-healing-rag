"""
safety/metrics.py
==================
Lightweight metrics collection for the platform.
Tracks counters, histograms, and gauges — persisted to JSONL.

Metrics collected:
  - requests_total           (counter, by agent)
  - request_latency_ms       (histogram)
  - confidence_score         (histogram)
  - self_heal_rate           (gauge)
  - verification_pass_rate   (gauge)
  - token_usage              (counter)
  - errors_total             (counter, by type)
"""

from __future__ import annotations
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    name: str
    value: float
    labels: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    metric_type: str = "counter"    # counter | histogram | gauge


class MetricsCollector:
    """
    Thread-safe in-memory metrics store with JSONL flush.
    """

    def __init__(self, output_path: str = "./data/logs/metrics.jsonl",
                 flush_interval: int = 30):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._points: list[MetricPoint] = []

    # ── Counter ───────────────────────────────────────────────────────────────
    def increment(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._counters[key] += value
            self._record(MetricPoint(name=name, value=value, labels=labels or {},
                                     metric_type="counter"))

    # ── Histogram ─────────────────────────────────────────────────────────────
    def observe(self, name: str, value: float, labels: dict | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._histograms[key].append(value)
            self._record(MetricPoint(name=name, value=value, labels=labels or {},
                                     metric_type="histogram"))

    # ── Gauge ─────────────────────────────────────────────────────────────────
    def set_gauge(self, name: str, value: float, labels: dict | None = None) -> None:
        with self._lock:
            key = self._key(name, labels)
            self._gauges[key] = value
            self._record(MetricPoint(name=name, value=value, labels=labels or {},
                                     metric_type="gauge"))

    # ── Convenience helpers ───────────────────────────────────────────────────
    def record_request(self, agent: str, latency_ms: float, confidence: float,
                       verified: bool, attempts: int, tokens: int = 0) -> None:
        self.increment("requests_total", labels={"agent": agent})
        self.observe("request_latency_ms", latency_ms, labels={"agent": agent})
        self.observe("confidence_score", confidence)
        self.increment("verification_pass_total" if verified else "verification_fail_total")
        if attempts > 1:
            self.increment("self_heal_triggered_total")
        if tokens:
            self.increment("token_usage_total", value=tokens, labels={"agent": agent})

    def record_error(self, error_type: str, agent: str = "unknown") -> None:
        self.increment("errors_total", labels={"type": error_type, "agent": agent})

    # ── Summary ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        with self._lock:
            out: dict = {"counters": {}, "histograms": {}, "gauges": dict(self._gauges)}
            for k, v in self._counters.items():
                out["counters"][k] = v
            for k, vals in self._histograms.items():
                if vals:
                    out["histograms"][k] = {
                        "count": len(vals),
                        "mean": round(sum(vals) / len(vals), 3),
                        "min": round(min(vals), 3),
                        "max": round(max(vals), 3),
                        "p95": round(sorted(vals)[int(len(vals) * 0.95)], 3),
                    }
            return out

    # ── Persistence ───────────────────────────────────────────────────────────
    def flush(self) -> None:
        with self._lock:
            points = list(self._points)
            self._points.clear()
        if not points:
            return
        try:
            with self.output_path.open("a") as f:
                for pt in points:
                    f.write(json.dumps(asdict(pt)) + "\n")
        except Exception as e:
            logger.error("Metrics flush failed: %s", e)

    def _record(self, point: MetricPoint) -> None:
        self._points.append(point)
        if len(self._points) >= 100:
            self.flush()

    @staticmethod
    def _key(name: str, labels: dict | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Module-level singleton
metrics = MetricsCollector()
