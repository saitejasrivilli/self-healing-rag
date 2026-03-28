"""
memory/memory_manager.py
=========================
Unified facade over all memory layers:
  - ShortTermMemory  (in-session working memory)
  - LongTermMemory   (cross-session semantic memory)
  - EpisodicMemory   (full episode logs)
  - VectorStore      (raw embedding access)

The MemoryManager is the single import that agent orchestration
needs to read/write across all memory tiers.
"""

from __future__ import annotations
import logging
import time
import uuid
from typing import Any, Optional, TYPE_CHECKING

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic_memory import EpisodicMemory, Episode
from memory.vector_store import VectorStore

if TYPE_CHECKING:
    from rag_pipeline import RAGResponse

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Single entry-point for all memory operations.
    Decides which tier to read from / write to based on context.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        episode_log: str = "./data/logs/episodes.jsonl",
        short_term_turns: int = 10,
        long_term_min_confidence: float = 0.75,
    ):
        self.short_term = ShortTermMemory(max_turns=short_term_turns)
        self.long_term  = LongTermMemory(persist_dir=persist_dir + "_memory")
        self.episodic   = EpisodicMemory(store_path=episode_log)
        self.vector     = VectorStore(persist_dir=persist_dir, collection_name="knowledge_base")
        self.long_term_min_confidence = long_term_min_confidence
        logger.info("MemoryManager initialised")

    # ── Write path ────────────────────────────────────────────────────────────
    def record_turn(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Add a conversation turn to short-term memory."""
        self.short_term.add_turn(role, content, metadata)

    def record_response(self, query: str, response: "RAGResponse") -> None:
        """
        After each pipeline response:
          1. Append to short-term history
          2. Record episode
          3. Promote to long-term if high confidence
        """
        self.short_term.add_turn("assistant", response.answer)
        self.short_term.cache_chunks(query, response.sources)

        episode = Episode(
            episode_id=str(uuid.uuid4())[:8],
            query=query,
            answer=response.answer,
            confidence=response.confidence,
            verified=response.verified,
            attempts=response.attempts,
            latency_ms=response.latency_ms,
            sources=[getattr(s, "source", "") for s in response.sources],
            query_used=response.query_used,
            reasoning=response.reasoning,
            timestamp=time.time(),
        )
        self.episodic.record(episode)

        # Promote high-confidence verified answers to long-term memory
        if response.verified and response.confidence >= self.long_term_min_confidence:
            self.long_term.store_qa(query, response.answer, response.confidence)
            logger.debug("Promoted to long-term memory: '%s...'", query[:60])

    # ── Read path ─────────────────────────────────────────────────────────────
    def build_context(self, query: str, top_k_long: int = 3) -> dict[str, Any]:
        """
        Assemble multi-tier context for the current query:
          - conversation_history: last 5 turns (short-term)
          - relevant_memories:    semantically similar long-term facts
          - similar_episodes:     past similar Q&A (for few-shot reference)
        """
        return {
            "conversation_history": self.short_term.format_history(last_n=5),
            "relevant_memories": self.long_term.recall(query, top_k=top_k_long),
            "similar_episodes": [
                {"query": ep.query, "answer": ep.answer, "confidence": ep.confidence}
                for ep in self.episodic.recall_similar(query, top_k=2)
            ],
        }

    def get_history(self, last_n: int = 10) -> list[dict]:
        return [
            {"role": t.role, "content": t.content}
            for t in self.short_term.get_turns(last_n)
        ]

    # ── Management ────────────────────────────────────────────────────────────
    def clear_session(self) -> None:
        """Reset short-term memory (e.g. on new session)."""
        self.short_term.clear()

    def stats(self) -> dict:
        return {
            "short_term": self.short_term.stats(),
            "long_term_entries": self.long_term.count(),
            "episodes": self.episodic.stats(),
            "vector_store": self.vector.stats(),
        }

    def export_training_data(self, min_confidence: float = 0.75) -> list[dict]:
        """Export verified episodes as SFT fine-tuning pairs."""
        return self.episodic.to_training_dataset(min_confidence)
