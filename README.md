# 🤖 Agentic AI Platform

[![CI](https://github.com/saitejasrivilli/agentic-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/saitejasrivilli/agentic-ai-platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **modular, production-grade agentic AI platform** built on the Self-Healing RAG foundation. Eight independent layers — orchestration, cognition, memory, tools, knowledge, LLM abstraction, and safety — compose into a fully observable, multi-turn agentic system.

---

## 🏗️ Architecture

```
agentic_ai_platform/
├── config/              Centralized configuration (YAML)
│   ├── base.yaml
│   ├── model_registry.yaml
│   ├── routing_policy.yaml
│   ├── environment.yaml
│   └── observability.yaml
│
├── orchestration/       Core control plane
│   ├── agent_router.py  Route queries → agents via routing_policy.yaml
│   ├── workflow_engine.py  Async self-healing RAG orchestrator
│   ├── task_graph.py    Async DAG executor (parallel pipeline steps)
│   └── state_manager.py Session state, multi-turn history
│
├── cognition/           Reasoning + planning layer
│   ├── planner.py       Decompose complex queries into sub-tasks
│   ├── reasoning_engine.py  CoT / self-ask reasoning modes
│   ├── reflection.py    Post-generation self-critique
│   └── decision_policy.py  ACCEPT / RETRY / ESCALATE logic
│
├── memory/              Memory abstraction layer
│   ├── short_term.py    In-session working memory (sliding window)
│   ├── long_term.py     Cross-session semantic memory (ChromaDB)
│   ├── vector_store.py  Unified vector store abstraction
│   ├── episodic_memory.py  Full episode logs + training export
│   └── memory_manager.py   Unified facade over all memory tiers
│
├── tools/               Tool execution layer
│   ├── tool_registry.py Central registry + @register_tool decorator
│   ├── api_tools.py     HTTP GET/POST, web search, fetch JSON
│   ├── db_tools.py      SQLite query, schema inspect, key-value store
│   └── external_actions.py  Composite actions (summarize_url, research_topic)
│
├── knowledge/           RAG layer
│   ├── retriever.py     Hybrid BM25 + ChromaDB + RRF fusion
│   ├── embeddings.py    HyDE query expansion
│   ├── db_tools.py      CrossEncoder reranker
│   └── external_actions.py  Pluggable chunking strategies
│
├── llm/                 Model abstraction layer
│   ├── llm_gateway.py   Core pipeline (generator, verifier, expander)
│   ├── prompt_templates.py  Versioned prompt library (7 templates)
│   ├── output_parser.py Robust JSON + text extraction
│   └── model_selector.py   Dynamic model selection by task type
│
├── safety/              Production-grade telemetry + guardrails
│   ├── tracing.py       Distributed tracing (JSONL / OTel-compatible)
│   ├── logging.py       Persistent observability logs
│   ├── metrics.py       Counters, histograms, gauges with p95
│   ├── token_monitor.py Per-request / per-minute token + cost tracking
│   ├── input_guard.py   Injection detection, PII redaction, length limits
│   ├── output_guard.py  Toxicity, uncertainty, PII checks on answers
│   ├── policy_engine.py Unified input/output policy + rate limiting
│   └── bias_detection.py   Gender/racial/political/sentiment bias flags
│
├── api/                 REST API layer
│   ├── routes.py        FastAPI endpoints (/ingest /query /metrics /history)
│   ├── schemas.py       Pydantic request/response models
│   └── middleware/
│       └── logging_middleware.py  Request logging + trace ID injection
│
├── apps/
│   ├── chatbot_ui/      Streamlit demo (streaming, HyDE, analytics tabs)
│   ├── admin_dashboard/ Monitoring dashboard (metrics, traces, safety flags)
│   └── cli/             Command-line interface (ingest/query/status/eval)
│
├── tests/
│   ├── unit/            Fast unit tests (no API key required)
│   ├── integration/     Multi-component pipeline tests
│   └── agent_tests/     Agent routing, task graph, state management
│
├── deployment/
│   ├── docker/          Dockerfile + docker-compose.yml
│   ├── k8s/             Kubernetes manifests (coming soon)
│   └── ci_cd/           GitHub Actions workflows
│
└── data/
    ├── raw/             Source documents
    ├── processed/       Golden Q&A dataset
    ├── embeddings/      Cached embeddings
    └── logs/            Observability JSONL logs
```

---

## ⚡ Quick Start

```bash
git clone https://github.com/saitejasrivilli/agentic-ai-platform
cd agentic-ai-platform
pip install -r requirements.txt
cp .env.example .env         # add GEMINI_API_KEY

make demo                    # Streamlit UI     → http://localhost:8501
make api                     # FastAPI docs     → http://localhost:8000/docs
make dashboard               # Admin dashboard  → http://localhost:8502
make cli                     # CLI help
```

---

## 🐳 Docker

```bash
cp .env.example .env
make docker                  # API :8000 + Streamlit :8501
make docker-down
```

---

## 📡 API

```bash
# Health
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest -F "files=@paper.pdf"

# Query (with session for multi-turn)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is LoRA?", "session_id": "user_123"}'

# Metrics
curl http://localhost:8000/metrics

# History
curl "http://localhost:8000/history?page=1&page_size=20"
```

---

## 🖥️ CLI

```bash
python apps/cli/main.py ingest ./data/raw
python apps/cli/main.py query "What is hybrid search?"
python apps/cli/main.py status
python apps/cli/main.py history --last 5
python apps/cli/main.py eval
python apps/cli/main.py clear-db
```

---

## 🧪 Testing

```bash
make test             # all tests + coverage
make test-unit        # unit only (fast, no API key)
make test-integration # integration tests
make test-agents      # agent routing + task graph + state
```

**Test coverage across:**
- `tests/unit/` — InputGuard, OutputGuard, BiasDetector, TokenMonitor, ShortTermMemory, EpisodicMemory, OutputParser, HyDE, chunking, observability, retriever
- `tests/integration/` — Full pipeline, decision policy, multi-turn conversation
- `tests/agent_tests/` — AgentRouter (14 tests), TaskGraph (10 tests), StateManager (15 tests)

---

## 🔑 Configuration

```bash
# .env
GEMINI_API_KEY=your_key      # free at ai.google.dev
TOP_K_RETRIEVE=10
TOP_K_RERANK=4
CONFIDENCE_THRESHOLD=0.65
MAX_RETRIES=3
ENABLE_HYDE=true
```

All pipeline parameters also configurable per-request via API body.

---

## 🛣️ Roadmap

- [ ] Kubernetes Helm chart (`deployment/k8s/`)
- [ ] OpenTelemetry export (Jaeger / Grafana Tempo)
- [ ] LiteLLM adapter (any LLM backend)
- [ ] Multi-agent collaboration (agent-to-agent tool calls)
- [ ] Streaming FastAPI endpoint (`/query/stream`)
- [ ] Fine-tuning export pipeline (episodic → SFT dataset)

---

## 👤 Author

**Sai Teja Srivilli** — ML/AI Engineer

[![GitHub](https://img.shields.io/badge/GitHub-saitejasrivilli-181717?logo=github)](https://github.com/saitejasrivilli)
[![Portfolio](https://img.shields.io/badge/Portfolio-saitejasrivilli.github.io-blue)](https://saitejasrivilli.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2601.00110-b31b1b)](https://arxiv.org/abs/2601.00110)

---

MIT © Sai Teja Srivilli
# self-healing-rag
