"""
tools/external_actions.py
==========================
Higher-level composite actions that combine multiple tools
(API + DB + LLM) into single callable units.

Examples:
  - summarize_url     : fetch URL → LLM summarize
  - answer_from_web   : web_search → RAG → answer
  - store_and_index   : fetch content → chunk → embed → store in vector DB
"""

from __future__ import annotations
import logging
import os
from typing import Optional

from tools.tool_registry import registry
from tools.api_tools import http_get, web_search

logger = logging.getLogger(__name__)


@registry.register_tool(
    name="summarize_url",
    description="Fetch a URL and return a concise summary of its content.",
    category="composite",
)
def summarize_url(url: str, max_length: int = 300) -> str:
    result = http_get(url)
    text = result.get("body", "")[:3000]
    if not text:
        return "Could not fetch content from URL."

    # Simple extractive summary (first N chars of readable content)
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    summary = " ".join(lines[:5])[:max_length]
    return summary or text[:max_length]


@registry.register_tool(
    name="research_topic",
    description="Search the web for a topic and return condensed findings.",
    category="composite",
)
def research_topic(query: str, num_results: int = 3) -> dict:
    results = web_search(query, num_results=num_results)
    snippets = [r["snippet"] for r in results if r.get("snippet")]
    return {
        "query": query,
        "num_results": len(results),
        "summary": " | ".join(snippets[:3]),
        "sources": [r.get("url", "") for r in results],
    }


@registry.register_tool(
    name="format_answer",
    description="Format a raw answer string into structured output (markdown or plain).",
    category="utility",
)
def format_answer(
    answer: str,
    format_type: str = "markdown",
    add_sources: Optional[list[str]] = None,
) -> str:
    if format_type == "plain":
        text = answer
        if add_sources:
            text += "\n\nSources: " + ", ".join(add_sources)
        return text

    # Markdown
    lines = [f"**Answer:** {answer}"]
    if add_sources:
        lines.append("\n**Sources:**")
        lines.extend(f"- {s}" for s in add_sources)
    return "\n".join(lines)


@registry.register_tool(
    name="chunk_and_store",
    description="Chunk raw text and upsert into the knowledge vector store.",
    category="composite",
)
def chunk_and_store(text: str, source_name: str, chunk_size: int = 512) -> int:
    """Split text into chunks and insert into the knowledge VectorStore."""
    from memory.vector_store import VectorStore
    from knowledge.embeddings import HyDEExpander  # re-uses embeddings infra

    store = VectorStore(collection_name="knowledge_base")
    words = text.split()
    step = chunk_size // 5   # ~100 word chunks
    chunks, ids, metas = [], [], []
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i : i + step])
        if not chunk_text.strip():
            continue
        chunk_id = f"{source_name}_{i}"
        chunks.append(chunk_text)
        ids.append(chunk_id)
        metas.append({"source": source_name, "offset": i})

    if chunks:
        store.upsert(ids=ids, texts=chunks, metadatas=metas)
    return len(chunks)
