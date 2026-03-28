"""
Self-Healing RAG Pipeline
Author: Sai Teja Srivilli
Architecture: Retriever → Reranker → LLM Generator → Verification Agent → Self-Healing Loop
"""

from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
@dataclass
class RAGConfig:
    # Retrieval
    collection_name: str = "self_healing_rag"
    top_k_retrieve: int = 10
    top_k_rerank: int = 4
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Generation
    model_name: str = "gemini-1.5-flash"          # swap for any HF model
    temperature: float = 0.2
    max_tokens: int = 512

    # Self-healing
    max_retries: int = 3
    confidence_threshold: float = 0.65            # below this → retry
    query_expansion_on_retry: bool = True


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    chunk_id: str


@dataclass
class RAGResponse:
    answer: str
    confidence: float
    sources: list[RetrievedChunk]
    attempts: int
    verified: bool
    reasoning: str
    latency_ms: float
    query_used: str                               # may differ from original after expansion


# ──────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────
class VectorRetriever:
    """Dense retrieval over ChromaDB with sentence-transformer embeddings."""

    def __init__(self, config: RAGConfig, persist_dir: str = "./chroma_db"):
        self.config = config
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-base-en-v1.5"
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("VectorRetriever ready — %d docs in collection", self.collection.count())

    def ingest(self, docs_path: str) -> int:
        """Chunk documents and upsert into ChromaDB. Returns chunks added."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        if os.path.isdir(docs_path):
            loader = DirectoryLoader(docs_path, glob="**/*.{txt,pdf,md}")
        elif docs_path.endswith(".pdf"):
            loader = PyPDFLoader(docs_path)
        else:
            loader = TextLoader(docs_path)

        raw_docs = loader.load()
        chunks = splitter.split_documents(raw_docs)

        ids, texts, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}_{hash(chunk.page_content) & 0xFFFF:04x}"
            ids.append(chunk_id)
            texts.append(chunk.page_content)
            metas.append({"source": chunk.metadata.get("source", "unknown"), "page": str(chunk.metadata.get("page", ""))})

        if ids:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
            logger.info("Ingested %d chunks from %s", len(ids), docs_path)
        return len(ids)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
        k = top_k or self.config.top_k_retrieve
        results = self.collection.query(query_texts=[query], n_results=min(k, self.collection.count() or 1))
        chunks = []
        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "unknown"),
                score=1 - dist,           # cosine distance → similarity
                chunk_id=cid,
            ))
        return chunks


# ──────────────────────────────────────────────
# Reranker
# ──────────────────────────────────────────────
class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs with a cross-encoder for precision."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoder reranker loaded: %s", model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 4) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return ranked[:top_k]


# ──────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, grounded AI assistant.
Answer ONLY using the provided context. If the context is insufficient, say so clearly.
Do not fabricate information. Be concise and cite which part of the context supports your answer."""

ANSWER_PROMPT = """Context:
{context}

Question: {query}

Provide a clear, factual answer grounded strictly in the context above."""


class LLMGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                config.model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_tokens,
                ),
            )
            self._backend = "gemini"
        else:
            # Fallback: use a small local HF model via transformers pipeline
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct",
                                  trust_remote_code=True, max_new_tokens=config.max_tokens)
            self._backend = "phi3"
        logger.info("LLMGenerator using backend: %s", self._backend)

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )
        prompt = ANSWER_PROMPT.format(context=context, query=query)

        if self._backend == "gemini":
            response = self.model.generate_content(prompt)
            return response.text.strip()
        else:
            full = f"{SYSTEM_PROMPT}\n\n{prompt}"
            out = self.pipe(full)[0]["generated_text"]
            return out[len(full):].strip()


# ──────────────────────────────────────────────
# Verification Agent
# ──────────────────────────────────────────────
VERIFY_PROMPT = """You are a strict factual verifier.

Given the retrieved context and the generated answer, assess:
1. Is every claim in the answer supported by the context?
2. Are there any hallucinations or unsupported statements?

Respond in this exact JSON format (no markdown):
{{
  "confidence": <float 0.0–1.0>,
  "verified": <true|false>,
  "reasoning": "<one sentence>"
}}"""


class VerificationAgent:
    def __init__(self, config: RAGConfig):
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                system_instruction=VERIFY_PROMPT,
                generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=200),
            )
            self._backend = "gemini"
        else:
            self._backend = "heuristic"

    def verify(self, query: str, answer: str, chunks: list[RetrievedChunk]) -> tuple[float, bool, str]:
        """Returns (confidence, verified, reasoning)."""
        if self._backend == "heuristic":
            return self._heuristic_verify(answer, chunks)

        context = "\n\n".join(c.text for c in chunks)
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer: {answer}\n\n"
            "Evaluate and respond with JSON only."
        )
        try:
            import json
            resp = self.model.generate_content(prompt)
            data = json.loads(resp.text.strip())
            return float(data["confidence"]), bool(data["verified"]), data["reasoning"]
        except Exception as e:
            logger.warning("Verification parse error: %s — falling back to heuristic", e)
            return self._heuristic_verify(answer, chunks)

    @staticmethod
    def _heuristic_verify(answer: str, chunks: list[RetrievedChunk]) -> tuple[float, bool, str]:
        """Simple token-overlap confidence when LLM verifier unavailable."""
        if not answer or not chunks:
            return 0.0, False, "Empty answer or no context."
        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(c.text for c in chunks).lower().split())
        overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        low_confidence_phrases = ["i don't know", "cannot determine", "not mentioned", "no information"]
        if any(p in answer.lower() for p in low_confidence_phrases):
            return 0.3, False, "Answer explicitly acknowledges insufficient context."
        confidence = min(overlap * 2.5, 1.0)
        return confidence, confidence >= 0.65, f"Token overlap confidence: {confidence:.2f}"


# ──────────────────────────────────────────────
# Query Expander (used in self-healing loop)
# ──────────────────────────────────────────────
EXPAND_PROMPT = """The following query failed to retrieve sufficient context.
Rephrase it with different keywords to improve retrieval. Return ONLY the rephrased query, nothing else.

Original query: {query}
Previous answer (low confidence): {answer}"""


class QueryExpander:
    def __init__(self, config: RAGConfig):
        self.config = config
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config=genai.GenerationConfig(temperature=0.7, max_output_tokens=100),
            )
            self._backend = "gemini"
        else:
            self._backend = "none"

    def expand(self, query: str, previous_answer: str) -> str:
        if self._backend == "none":
            return query + " explain in detail"
        prompt = EXPAND_PROMPT.format(query=query, answer=previous_answer)
        try:
            resp = self.model.generate_content(prompt)
            expanded = resp.text.strip()
            logger.info("Query expanded: '%s' → '%s'", query, expanded)
            return expanded
        except Exception:
            return query


# ──────────────────────────────────────────────
# Self-Healing RAG — Orchestrator
# ──────────────────────────────────────────────
class SelfHealingRAG:
    """
    Full pipeline:
      Retriever → Reranker → LLM Generator → Verification Agent
      └─ if confidence < threshold: QueryExpander → retry (up to max_retries)
    """

    def __init__(self, config: Optional[RAGConfig] = None, persist_dir: str = "./chroma_db"):
        self.config = config or RAGConfig()
        self.retriever = VectorRetriever(self.config, persist_dir)
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator(self.config)
        self.verifier = VerificationAgent(self.config)
        self.expander = QueryExpander(self.config)

    def ingest(self, path: str) -> int:
        return self.retriever.ingest(path)

    def query(self, user_query: str) -> RAGResponse:
        start = time.time()
        current_query = user_query
        attempt = 0
        best_response: Optional[RAGResponse] = None

        while attempt < self.config.max_retries:
            attempt += 1
            logger.info("Attempt %d — query: %s", attempt, current_query)

            # 1. Retrieve
            chunks = self.retriever.retrieve(current_query)
            if not chunks:
                logger.warning("No chunks retrieved.")
                break

            # 2. Rerank
            ranked = self.reranker.rerank(current_query, chunks, self.config.top_k_rerank)

            # 3. Generate
            answer = self.generator.generate(current_query, ranked)

            # 4. Verify
            confidence, verified, reasoning = self.verifier.verify(current_query, answer, ranked)
            logger.info("Confidence: %.2f | Verified: %s", confidence, verified)

            response = RAGResponse(
                answer=answer,
                confidence=confidence,
                sources=ranked,
                attempts=attempt,
                verified=verified,
                reasoning=reasoning,
                latency_ms=(time.time() - start) * 1000,
                query_used=current_query,
            )

            if best_response is None or confidence > best_response.confidence:
                best_response = response

            if verified and confidence >= self.config.confidence_threshold:
                break

            # Self-healing: expand query and retry
            if attempt < self.config.max_retries and self.config.query_expansion_on_retry:
                current_query = self.expander.expand(user_query, answer)

        best_response.latency_ms = (time.time() - start) * 1000
        return best_response
