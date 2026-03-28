"""
api/schemas.py
==============
Pydantic schemas for the FastAPI REST layer.
Single source of truth for all request/response shapes.
"""

from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Requests ─────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversations")
    top_k_retrieve: Optional[int] = Field(None, ge=1, le=50)
    top_k_rerank: Optional[int] = Field(None, ge=1, le=20)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_retries: Optional[int] = Field(None, ge=1, le=5)
    enable_hyde: Optional[bool] = None
    reasoning_mode: Optional[str] = Field(None, description="direct | cot | self_ask")
    client_id: str = Field("default", description="Client identifier for rate limiting")


class IngestRequest(BaseModel):
    source_name: Optional[str] = Field(None, description="Label for the ingested content")
    chunk_size: int = Field(512, ge=128, le=2048)
    chunk_overlap: int = Field(64, ge=0, le=256)


class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)


# ── Responses ─────────────────────────────────────────────────────────────────
class SourceOut(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: str
    retrieval_method: str = "hybrid"


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    verified: bool
    attempts: int
    latency_ms: float
    query_used: str
    reasoning: str
    session_id: Optional[str] = None
    sources: list[SourceOut] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    chunks_indexed: int
    files_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    pipeline_ready: bool
    collection_size: int
    gemini_api_configured: bool
    active_sessions: int = 0


class MetricsResponse(BaseModel):
    status: str
    metrics: dict[str, Any]
    token_usage: dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: int = 500
