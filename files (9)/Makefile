# Agentic AI Platform — Makefile
.PHONY: help install demo api dashboard cli eval test docker clean lint

PYTHON := python3
PIP    := pip3

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*##"}{printf "\033[36m%-20s\033[0m %s\n",$$1,$$2}'

install:  ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

demo:  ## Streamlit chatbot UI (port 8501)
	streamlit run apps/chatbot_ui/app.py --server.port 8501

api:   ## FastAPI REST server (port 8000)
	uvicorn api.routes:app --host 0.0.0.0 --port 8000 --reload

dashboard:  ## Admin monitoring dashboard (port 8502)
	streamlit run apps/admin_dashboard/dashboard.py --server.port 8502

cli:   ## CLI help
	$(PYTHON) apps/cli/main.py --help

ingest:  ## Ingest sample knowledge base
	$(PYTHON) apps/cli/main.py ingest ./data/raw

status:  ## Platform status
	$(PYTHON) apps/cli/main.py status

eval:  ## Run RAGAS evaluation
	$(PYTHON) examples/evaluate.py \
		--docs ./data/raw \
		--golden ./data/processed/qa_pairs.json \
		--out ./data/logs/eval_results.json

test:  ## All tests with coverage
	pytest tests/ -v --tb=short \
		--cov=orchestration --cov=cognition --cov=memory \
		--cov=tools --cov=llm --cov=safety --cov-report=term-missing

test-unit:  ## Unit tests only
	pytest tests/unit/ -v --tb=short

test-integration:  ## Integration tests
	pytest tests/integration/ -v --tb=short

test-agents:  ## Agent tests
	pytest tests/agent_tests/ -v --tb=short

docker:  ## Full stack via docker compose
	docker compose -f deployment/docker/docker-compose.yml up --build

docker-down:  ## Stop all services
	docker compose -f deployment/docker/docker-compose.yml down

lint:  ## Lint with ruff
	ruff check orchestration/ cognition/ memory/ tools/ llm/ safety/ api/ apps/ --ignore E501

format:  ## Format with black
	black orchestration/ cognition/ memory/ tools/ llm/ safety/ api/ apps/ tests/

clean:  ## Remove caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ chroma_db_eval_*
	@echo "Cleaned."

clean-db:  ## Reset ChromaDB
	rm -rf chroma_db/ chroma_db_memory/
	@echo "ChromaDB cleared."

benchmark:  ## Run pipeline benchmark (p50/p95/p99 latency)
	$(PYTHON) scripts/benchmark.py --docs ./data/raw --runs 9 \
		--out ./data/logs/benchmark.json --configs full no_healing

fine-tune:  ## Export SFT dataset from episodic memory
	$(PYTHON) scripts/finetune_export.py \
		--min-confidence 0.75 --format chatml \
		--out ./data/sft_dataset.jsonl

stream-test:  ## Test the streaming endpoint (requires running API)
	curl -N -X POST http://localhost:8000/stream/query \
		-H "Content-Type: application/json" \
		-d '{"query": "What is hybrid search?"}'

linkedin:  ## Print LinkedIn post draft
	$(PYTHON) scripts/linkedin_post.py
