"""
memory/long_term.py
====================
Persistent long-term memory backed by ChromaDB.
Stores important facts, user preferences, and past Q&A pairs
that should survive session boundaries.

Use cases:
  - "The user prefers concise answers" (preference)
  - "The answer to Q was X" (cached Q&A)
  - "Document Y was ingested on date Z" (provenance)
"""

from __future__ import annotations
import json
import logging
import time
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class LongTermMemory:
    """
    ChromaDB-backed persistent memory with semantic retrieval.
    Entries are stored as embedding vectors for fuzzy recall.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db_memory",
        collection_name: str = "long_term_memory",
        embed_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("LongTermMemory: %d entries loaded", self.collection.count())

    def store(self, key: str, content: str, memory_type: str = "fact",
              metadata: dict | None = None) -> None:
        """Store a memory entry (idempotent — upserts by key)."""
        meta = {
            "type": memory_type,
            "stored_at": str(time.time()),
            **(metadata or {}),
        }
        self.collection.upsert(ids=[key], documents=[content], metadatas=[meta])
        logger.debug("LongTermMemory: stored '%s' (%s)", key, memory_type)

    def recall(self, query: str, top_k: int = 5,
               memory_type: Optional[str] = None) -> list[dict]:
        """Recall memories semantically similar to query."""
        if self.collection.count() == 0:
            return []
        where = {"type": memory_type} if memory_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.collection.count()),
            where=where,
        )
        memories = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            memories.append({
                "id": cid,
                "content": doc,
                "type": meta.get("type"),
                "similarity": round(1 - dist, 4),
                "stored_at": meta.get("stored_at"),
            })
        return memories

    def forget(self, key: str) -> bool:
        """Delete a specific memory by key."""
        try:
            self.collection.delete(ids=[key])
            logger.info("LongTermMemory: deleted '%s'", key)
            return True
        except Exception:
            return False

    def store_qa(self, question: str, answer: str, confidence: float) -> None:
        """Convenience: store a verified Q&A pair."""
        key = f"qa_{hash(question) & 0xFFFF:04x}"
        self.store(
            key=key,
            content=f"Q: {question}\nA: {answer}",
            memory_type="qa_pair",
            metadata={"confidence": str(confidence)},
        )

    def count(self) -> int:
        return self.collection.count()
