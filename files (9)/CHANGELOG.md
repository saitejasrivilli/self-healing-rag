# Changelog

All notable changes to the Agentic AI Platform are documented here.
Follows [Keep a Changelog](https://keepachangelog.com) and [Semantic Versioning](https://semver.org).

---

## [1.3.0] — 2025-06-01

### Added
- **Streaming SSE endpoint** (`POST /stream/query`) — yields `stage`, `token`, and `done` events via Server-Sent Events for real-time word-by-word streaming
- **Multi-agent orchestration** (`orchestration/multi_agent.py`) — `RAGAgent`, `ToolAgent`, `SynthesisAgent`, and `MultiAgentCoordinator` running agents in parallel via `asyncio.gather`
- **LiteLLM adapter** (`llm/litellm_adapter.py`) — swap any LLM provider (Gemini, OpenAI, Anthropic, Ollama, Groq, Azure) via `LLM_MODEL` env var with automatic fallback chain
- **Fine-tuning export pipeline** (`scripts/finetune_export.py`) — exports verified episodes as Alpaca/ChatML/raw-pairs JSONL; optional QLoRA training via HuggingFace TRL on A30 cluster
- **Kubernetes + Helm chart** — full K8s deployment with HPA (2–8 replicas), PVCs (ChromaDB 10Gi), TLS Ingress, NetworkPolicy, Kustomize overlays
- **CI/CD pipeline** (`deployment/ci_cd/pipeline.yml`) — test → build multi-arch Docker → deploy staging (main) → deploy prod (release tag)
- **Benchmark script** (`scripts/benchmark.py`) — p50/p95/p99 latency + quality metrics across 4 pipeline configurations
- **Technical writeup PDF** — 7-section arXiv-style paper with ablation study and results tables
- 35+ new tests covering streaming, multi-agent, LiteLLM adapter, model selector, and prompt templates

### Changed
- `api/routes.py` now mounts the streaming router at `/stream`
- `README.md` updated with Helm, LiteLLM, streaming, and multi-agent documentation
- `Makefile` updated with `benchmark`, `stream-test`, and `fine-tune` targets

---

## [1.2.0] — 2025-05-15

### Added
- **Agentic AI Platform** — 8-layer modular architecture matching the Enhanced Agentic AI Project Architecture diagram
- **Orchestration layer** — `AgentRouter` (policy-driven routing), `TaskGraph` (async DAG executor), `StateManager` (TTL-based session state with disk persistence)
- **Cognition layer** — `QueryPlanner` (LLM or heuristic decomposition), `ReasoningEngine` (direct/CoT/self-ask), `ReflectionAgent` (post-generation critique), `DecisionPolicy` (ACCEPT/RETRY/ESCALATE)
- **Memory layer** — `ShortTermMemory` (sliding window), `LongTermMemory` (ChromaDB semantic), `EpisodicMemory` (JSONL episodes + training export), `VectorStore` (unified abstraction), `MemoryManager` (facade)
- **Tools layer** — `ToolRegistry` (@decorator pattern, singleton), `APITools` (HTTP, web_search, webhooks), `DBTools` (SQLite, schema inspect, KV store), `ExternalActions` (composite actions)
- **LLM layer** — `PromptTemplates` (7 versioned templates), `OutputParser` (robust JSON extraction), `ModelSelector` (task-based dynamic selection)
- **Safety layer** — `Tracer` (JSONL spans), `MetricsCollector` (counters/histograms/gauges/p95), `TokenMonitor` (cost estimation + alerts), `InputGuard` (injection + PII), `OutputGuard` (toxicity + uncertainty), `PolicyEngine` (rate limiting), `BiasDetector`
- **Admin dashboard** (`apps/admin_dashboard/dashboard.py`) — Plotly charts for confidence, quality breakdown, traces, safety flags
- **CLI** (`apps/cli/main.py`) — ingest/query/status/history/eval/clear-db commands
- **Config layer** — 5 YAML files for base, model registry, routing policy, environment, observability
- 130+ tests across unit, integration, and agent test suites
- `pytest.ini` with `slow`/`unit`/`integration`/`agent` markers
- `.gitignore`, `LICENSE` (MIT), `Makefile` with 20+ targets

### Changed
- Restructured from flat `src/` to 8-module package hierarchy
- `ChromaDB` now accessed via `memory/vector_store.py` abstraction throughout

---

## [1.1.0] — 2025-05-01

### Added
- **Hybrid BM25 + Dense retrieval** with Reciprocal Rank Fusion (RRF, k=60)
- **HyDE** (Hypothetical Document Embeddings) — generate hypothetical answer, embed for retrieval
- **Async pipeline** (`AsyncSelfHealingRAG`) — `asyncio` + `ThreadPoolExecutor` for concurrent requests
- **FastAPI REST API** — `/ingest`, `/query`, `/metrics`, `/history`, `/health`, `/collection` endpoints
- **RAGAS evaluation harness** (`scripts/evaluate.py`) — Standard RAG vs Self-Healing RAG on golden dataset
- **Streaming Streamlit UI** — word-by-word streaming, HyDE toggle, analytics tab with 3 Plotly charts
- **Evaluation tab** in Streamlit — run RAGAS and display results table inline
- **Pluggable chunking** (`chunker.py`) — `RecursiveCharacter` and `Semantic` strategies, toggled from sidebar
- **`PipelineObserver`** — persistent JSONL logging with `summary_stats()`
- **Docker Compose** — multi-stage Dockerfile, shared volumes, eval runner profile
- **GitHub Actions CI** — Python 3.10 + 3.11 matrix, coverage upload
- 36 tests across 4 test files, `pytest.ini`
- `.gitignore`, `LICENSE` (MIT), `Makefile`, `.streamlit/config.toml`
- Golden dataset: 8 Q&A pairs in `data/golden_dataset/qa_pairs.json`

### Changed
- `VectorRetriever` replaced by `HybridRetriever` (BM25 + ChromaDB + RRF)
- `RAGResponse.query_used` field added (tracks expanded queries)
- `CrossEncoderReranker` extracted to `reranker.py`

---

## [1.0.0] — 2025-04-15

### Added
- Initial Self-Healing RAG implementation
- `VectorRetriever` — ChromaDB + BGE-base-en-v1.5 embeddings
- `CrossEncoderReranker` — ms-marco-MiniLM-L-6-v2
- `LLMGenerator` — Gemini 1.5 Flash with grounded system prompt
- `VerificationAgent` — LLM confidence scoring with JSON output + heuristic fallback
- `QueryExpander` — LLM-based query reformulation for retry
- `SelfHealingRAG` — full synchronous orchestrator with retry loop
- Streamlit demo with document upload, pipeline progress, source viewer
- `RAGConfig` dataclass for centralized configuration
- `requirements.txt` with ChromaDB, sentence-transformers, LangChain, Gemini
