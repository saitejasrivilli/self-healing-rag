# Contributing to Agentic AI Platform

Thank you for your interest in contributing. This document covers how to set up the dev environment, run tests, and submit changes.

---

## Development Setup

```bash
git clone https://github.com/saitejasrivilli/agentic-ai-platform
cd agentic-ai-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install ruff black pytest pytest-cov pytest-asyncio

# Copy env and add your Gemini key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_key_here
```

---

## Running Tests

```bash
make test             # all tests + coverage report
make test-unit        # fast unit tests only (no API key needed)
make test-integration # integration tests
make test-agents      # agent orchestration tests
```

Tests are organized into three suites:

| Suite | Location | Notes |
|---|---|---|
| `unit` | `tests/unit/` | No API key, no external deps, runs in <30s |
| `integration` | `tests/integration/` | No API key, tests full pipeline with mocks |
| `agent` | `tests/agent_tests/` | No API key, tests routing/DAG/state |

To run a specific file:
```bash
pytest tests/unit/test_safety.py -v
```

To run with a specific marker:
```bash
pytest -m unit -v
pytest -m "not slow" -v
```

---

## Code Style

We use **ruff** for linting and **black** for formatting:

```bash
make lint    # ruff check
make format  # black format
```

Key conventions:
- All public classes and functions have docstrings
- New modules include a module-level docstring explaining purpose and usage
- Type hints on all public function signatures
- Dataclasses for structured data (avoid raw dicts for return types)
- `logger = logging.getLogger(__name__)` at module level

---

## Adding a New Tool

1. Create your function in `tools/api_tools.py`, `tools/db_tools.py`, or a new file
2. Register it with the `@registry.register_tool(...)` decorator
3. Add tests in `tests/unit/test_tools.py` (or create it)
4. Update `tools/__init__.py` if exporting

```python
# Example: adding a new tool
from tools.tool_registry import registry

@registry.register_tool(
    name="my_new_tool",
    description="What this tool does in one sentence.",
    category="api",
)
def my_new_tool(param1: str, param2: int = 10) -> dict:
    """Full docstring here."""
    return {"result": f"{param1} * {param2}"}
```

---

## Adding a New LLM Prompt Template

Add to `llm/prompt_templates.py` using the `register()` function:

```python
MY_TEMPLATE = register(PromptTemplate(
    name="my_template",
    description="What this prompt does.",
    system="You are a ...",
    user="Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
    version="1.0",
))
```

Placeholders use Python's `str.format()` syntax — document all required variables in the description.

---

## Adding a New Safety Check

1. Add pattern lists and logic to `safety/input_guard.py` or `safety/output_guard.py`
2. For a new guard type, create `safety/my_guard.py` following the existing pattern
3. Register it in `safety/policy_engine.py`
4. Add tests in `tests/unit/test_safety.py`

---

## Submitting a Pull Request

1. **Branch naming:** `feat/description`, `fix/description`, `docs/description`, `test/description`
2. **Commit messages:** Follow [Conventional Commits](https://conventionalcommits.org):
   - `feat: add streaming SSE endpoint`
   - `fix: handle empty ChromaDB collection in retriever`
   - `docs: update README with Helm deployment steps`
   - `test: add multi-agent coordinator tests`
3. **PR description** should include: what changed, why, how to test it
4. All tests must pass: `make test`
5. Lint must pass: `make lint`
6. Update `CHANGELOG.md` under the `[Unreleased]` section

---

## Architecture Decisions

Key decisions documented in the codebase:

| Decision | Rationale |
|---|---|
| RRF for fusion | Score-agnostic; no normalization needed between BM25 and cosine |
| CrossEncoder reranker | Higher precision than bi-encoder for top-k selection |
| HyDE | Bridges query-document vocabulary gap for complex queries |
| JSONL observability | Simple, appendable, parseable; no external deps |
| Dataclasses for results | Type safety + IDE autocomplete vs raw dicts |
| ThreadPoolExecutor for CPU tasks | Avoids blocking the async event loop for reranker inference |

---

## Questions?

Open an issue or reach out via [GitHub](https://github.com/saitejasrivilli).
