"""
memory/vector_store.py
=======================
Unified vector store abstraction layer.
Wraps ChromaDB with a clean interface shared across
knowledge retrieval AND long-term memory.

Supports:
  - Named collections (knowledge_base, long_term_memory, episodes)
  - Batch upsert / delete
  - Metadata filtering
  - Collection stats
"""

from __future__ import annotations
import logging
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


class VectorStore:
    """
    Thin wrapper over ChromaDB that provides a consistent API
    for all vector storage needs in the platform.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "default",
        embed_model: str = DEFAULT_EMBED_MODEL,
        distance_metric: str = "cosine",
    ):
        self.collection_name = collection_name
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embed_model
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info("VectorStore '%s': %d docs", collection_name, self.count())

    # ── Write ─────────────────────────────────────────────────────────────────
    def upsert(self, ids: list[str], texts: list[str],
               metadatas: list[dict] | None = None) -> None:
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{} for _ in ids],
        )
        logger.debug("VectorStore '%s': upserted %d docs", self.collection_name, len(ids))

    def delete(self, ids: list[str]) -> None:
        self.collection.delete(ids=ids)

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.ef,
        )
        logger.info("VectorStore '%s': reset", self.collection_name)

    # ── Read ──────────────────────────────────────────────────────────────────
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Returns list of dicts:
          {"id": str, "text": str, "metadata": dict, "score": float}
        """
        n = min(top_k, self.count())
        if n == 0:
            return []
        kwargs: dict = {"query_texts": [query_text], "n_results": n}
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        output = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            output.append({
                "id": cid,
                "text": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),
            })
        return output

    def get_all(self, limit: int = 1000) -> list[dict]:
        result = self.collection.get(limit=limit, include=["documents", "metadatas"])
        return [
            {"id": cid, "text": doc, "metadata": meta}
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    # ── Stats ─────────────────────────────────────────────────────────────────
    def count(self) -> int:
        return self.collection.count()

    def stats(self) -> dict:
        return {
            "collection": self.collection_name,
            "doc_count": self.count(),
        }
