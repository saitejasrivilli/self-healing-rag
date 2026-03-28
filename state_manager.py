"""
orchestration/state_manager.py
================================
Manages conversation and session state across multi-turn agent interactions.
Thread-safe in-memory store with optional JSON persistence.

State schema:
  session_id → {
    "history":    [{"role": "user"|"assistant", "content": "..."}],
    "context":    {...}   # arbitrary key-value metadata
    "created_at": float,
    "updated_at": float,
  }
"""

from __future__ import annotations
import json
import logging
import os
import threading
import time
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class StateManager:
    """
    Thread-safe in-memory session state with optional disk persistence.
    """

    def __init__(self, persist_path: Optional[str] = None, ttl_seconds: int = 3600):
        self._store: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._persist_path = Path(persist_path) if persist_path else None
        self._ttl = ttl_seconds

        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # ── Session lifecycle ─────────────────────────────────────────────────────
    def create_session(self, session_id: str, context: dict | None = None) -> str:
        with self._lock:
            self._store[session_id] = {
                "history": [],
                "context": context or {},
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        logger.info("Session created: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            session = self._store.get(session_id)
            if session and self._is_expired(session):
                self.delete_session(session_id)
                return None
            return session

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._store
            self._store.pop(session_id, None)
        if existed:
            logger.info("Session deleted: %s", session_id)
        return existed

    # ── History management ────────────────────────────────────────────────────
    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            if session_id not in self._store:
                self.create_session(session_id)
            self._store[session_id]["history"].append({"role": role, "content": content})
            self._store[session_id]["updated_at"] = time.time()

    def get_history(self, session_id: str, last_n: int = 10) -> list[dict]:
        session = self.get_session(session_id)
        if not session:
            return []
        return session["history"][-last_n:]

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._store:
                self._store[session_id]["history"] = []
                self._store[session_id]["updated_at"] = time.time()

    # ── Context metadata ──────────────────────────────────────────────────────
    def set_context(self, session_id: str, key: str, value: Any) -> None:
        with self._lock:
            if session_id not in self._store:
                self.create_session(session_id)
            self._store[session_id]["context"][key] = value
            self._store[session_id]["updated_at"] = time.time()

    def get_context(self, session_id: str, key: str, default: Any = None) -> Any:
        session = self.get_session(session_id)
        if not session:
            return default
        return session["context"].get(key, default)

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_to_disk(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = dict(self._store)
        with open(self._persist_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("State persisted to %s (%d sessions)", self._persist_path, len(data))

    def _load_from_disk(self) -> None:
        try:
            with open(self._persist_path) as f:
                self._store = json.load(f)
            logger.info("State loaded from %s (%d sessions)", self._persist_path, len(self._store))
        except Exception as e:
            logger.warning("Failed to load state from disk: %s", e)

    # ── Utils ─────────────────────────────────────────────────────────────────
    def _is_expired(self, session: dict) -> bool:
        return (time.time() - session["updated_at"]) > self._ttl

    def active_sessions(self) -> int:
        with self._lock:
            return sum(1 for s in self._store.values() if not self._is_expired(s))

    def purge_expired(self) -> int:
        with self._lock:
            expired = [sid for sid, s in self._store.items() if self._is_expired(s)]
            for sid in expired:
                del self._store[sid]
        if expired:
            logger.info("Purged %d expired sessions", len(expired))
        return len(expired)
