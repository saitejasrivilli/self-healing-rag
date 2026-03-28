"""
memory/short_term.py
=====================
In-session short-term memory — stores the last N conversation turns
and retrieved chunks so the agent can reference recent context
without re-retrieving from the vector store.

Think of this as the agent's "working memory" or context window manager.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    role: str            # "user" | "assistant" | "system" | "retrieval"
    content: str
    metadata: dict = field(default_factory=dict)
    turn: int = 0


class ShortTermMemory:
    """
    Fixed-size sliding window of conversation turns + retrieved chunks.
    Automatically evicts oldest entries when capacity is exceeded.
    """

    def __init__(self, max_turns: int = 10, max_chunks: int = 20):
        self.max_turns = max_turns
        self.max_chunks = max_chunks
        self._turns: deque[MemoryEntry] = deque(maxlen=max_turns)
        self._chunks: deque[MemoryEntry] = deque(maxlen=max_chunks)
        self._turn_counter = 0

    # ── Turns ─────────────────────────────────────────────────────────────────
    def add_turn(self, role: str, content: str, metadata: dict | None = None) -> None:
        self._turn_counter += 1
        entry = MemoryEntry(role=role, content=content,
                            metadata=metadata or {}, turn=self._turn_counter)
        self._turns.append(entry)
        logger.debug("ShortTermMemory: added turn %d (%s)", self._turn_counter, role)

    def get_turns(self, last_n: Optional[int] = None) -> list[MemoryEntry]:
        turns = list(self._turns)
        if last_n:
            return turns[-last_n:]
        return turns

    def format_history(self, last_n: int = 5) -> str:
        """Format last N turns as a conversation string for LLM context."""
        turns = self.get_turns(last_n)
        lines = []
        for t in turns:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.content}")
        return "\n".join(lines)

    # ── Retrieved chunks cache ────────────────────────────────────────────────
    def cache_chunks(self, query: str, chunks: list[Any]) -> None:
        for chunk in chunks:
            self._chunks.append(MemoryEntry(
                role="retrieval",
                content=chunk.text if hasattr(chunk, "text") else str(chunk),
                metadata={"query": query, "source": getattr(chunk, "source", "unknown")},
                turn=self._turn_counter,
            ))

    def get_cached_chunks(self, query: str | None = None) -> list[MemoryEntry]:
        if query is None:
            return list(self._chunks)
        return [c for c in self._chunks if c.metadata.get("query") == query]

    # ── Utils ─────────────────────────────────────────────────────────────────
    def clear(self) -> None:
        self._turns.clear()
        self._chunks.clear()
        self._turn_counter = 0

    def stats(self) -> dict:
        return {
            "turns": len(self._turns),
            "cached_chunks": len(self._chunks),
            "total_turns_seen": self._turn_counter,
        }
