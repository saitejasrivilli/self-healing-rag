"""
safety/tracing.py
==================
Distributed tracing for the agentic pipeline.
Each request gets a trace_id; each pipeline step gets a span.

Output: JSONL (default), compatible with OpenTelemetry export.

Trace schema:
  {
    "trace_id": "abc123",
    "span_id": "def456",
    "parent_span_id": null,
    "name": "rag_pipeline.retrieve",
    "start_time": 1234567890.123,
    "end_time": 1234567890.456,
    "duration_ms": 333.0,
    "status": "ok" | "error",
    "attributes": {...}
  }
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Span:
    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "ok"
    parent_span_id: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    error: Optional[str] = None

    def finish(self, error: str | None = None) -> None:
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = "error" if error else "ok"
        self.error = error


class Tracer:
    def __init__(self, output_path: str = "./data/logs/traces.jsonl"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_spans: dict[str, Span] = {}

    def new_trace(self) -> str:
        return uuid.uuid4().hex[:12]

    def start_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        attributes: dict | None = None,
    ) -> Span:
        span = Span(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:8],
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        self._active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span, error: str | None = None) -> None:
        span.finish(error)
        self._active_spans.pop(span.span_id, None)
        self._write(span)

    def _write(self, span: Span) -> None:
        try:
            with self.output_path.open("a") as f:
                f.write(json.dumps(asdict(span)) + "\n")
        except Exception as e:
            logger.error("Tracer write failed: %s", e)

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        attributes: dict | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager for automatic span lifecycle."""
        s = self.start_span(name, trace_id, parent_span_id, attributes)
        try:
            yield s
        except Exception as e:
            self.finish_span(s, error=str(e))
            raise
        else:
            self.finish_span(s)

    def load_trace(self, trace_id: str) -> list[dict]:
        """Load all spans for a given trace_id."""
        if not self.output_path.exists():
            return []
        spans = []
        with self.output_path.open() as f:
            for line in f:
                try:
                    s = json.loads(line)
                    if s.get("trace_id") == trace_id:
                        spans.append(s)
                except Exception:
                    pass
        return sorted(spans, key=lambda s: s["start_time"])


# Module-level singleton
tracer = Tracer()
