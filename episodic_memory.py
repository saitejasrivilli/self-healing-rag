"""
memory/episodic_memory.py
==========================
Episodic memory stores complete interaction episodes —
each episode is a full query→retrieval→answer→verification cycle
with its metadata.

Enables:
  - Learning from past failures (which queries triggered self-healing?)
  - Identifying recurring difficult topics
  - Building a training dataset from high-confidence episodes
  - Temporal reasoning ("last time you asked about X, the answer was...")
"""

from __future__ import annotations
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    episode_id: str
    query: str
    answer: str
    confidence: float
    verified: bool
    attempts: int
    latency_ms: float
    sources: list[str]          # source filenames
    query_used: str             # may differ from query (HyDE / expansion)
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class EpisodicMemory:
    """
    Append-only JSONL store of interaction episodes with in-memory index.
    Supports filtering by confidence, verification status, and tags.
    """

    def __init__(self, store_path: str = "./data/logs/episodes.jsonl"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._episodes: list[Episode] = []
        self._load()

    def record(self, episode: Episode) -> None:
        """Append an episode to memory."""
        self._episodes.append(episode)
        with self.store_path.open("a") as f:
            f.write(json.dumps(episode.to_dict()) + "\n")
        logger.debug("Episode recorded: %s (conf=%.2f)", episode.episode_id, episode.confidence)

    def recall_similar(self, query: str, top_k: int = 5) -> list[Episode]:
        """Simple token overlap recall (no vector DB needed)."""
        q_tokens = set(query.lower().split())
        scored = []
        for ep in self._episodes:
            ep_tokens = set(ep.query.lower().split())
            overlap = len(q_tokens & ep_tokens) / max(len(q_tokens), 1)
            scored.append((overlap, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def recall_failures(self, min_attempts: int = 2) -> list[Episode]:
        """Return episodes that triggered self-healing."""
        return [ep for ep in self._episodes if ep.attempts >= min_attempts]

    def recall_verified(self, min_confidence: float = 0.8) -> list[Episode]:
        """Return high-confidence verified episodes (for dataset building)."""
        return [ep for ep in self._episodes
                if ep.verified and ep.confidence >= min_confidence]

    def to_training_dataset(self, min_confidence: float = 0.75) -> list[dict]:
        """Export high-confidence episodes as SFT training pairs."""
        episodes = self.recall_verified(min_confidence)
        return [{"prompt": ep.query, "completion": ep.answer} for ep in episodes]

    def stats(self) -> dict:
        n = len(self._episodes)
        if n == 0:
            return {"total": 0}
        return {
            "total": n,
            "avg_confidence": round(sum(e.confidence for e in self._episodes) / n, 3),
            "verified_rate": round(sum(1 for e in self._episodes if e.verified) / n, 3),
            "self_heal_rate": round(sum(1 for e in self._episodes if e.attempts > 1) / n, 3),
            "avg_latency_ms": round(sum(e.latency_ms for e in self._episodes) / n, 1),
        }

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        with self.store_path.open() as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self._episodes.append(Episode(**data))
                except Exception:
                    pass
        logger.info("EpisodicMemory: loaded %d episodes", len(self._episodes))
