"""
api/middleware/logging_middleware.py
=====================================
FastAPI middleware that logs every request/response
with trace_id, latency, and status code.
"""

from __future__ import annotations
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:12])
        start = time.time()

        # Attach trace_id to request state for downstream use
        request.state.trace_id = trace_id

        response = await call_next(request)
        latency_ms = round((time.time() - start) * 1000, 1)

        logger.info(
            "%s %s %d %.0fms trace=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            trace_id,
        )

        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Latency-Ms"] = str(latency_ms)
        return response
