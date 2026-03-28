"""
tools/tool_registry.py
=======================
Central registry for all agent-callable tools.
Tools are registered by name and discovered by the agent at runtime.

Registry pattern:
  - @register_tool decorator adds a function to the registry
  - AgentRouter / WorkflowEngine calls registry.invoke(tool_name, **kwargs)
  - Each tool returns a ToolResult with output + metadata

Built-in tools: web_search, db_query, calculator, document_fetch
Custom tools: registered via register_tool()
"""

from __future__ import annotations
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    output: Any
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable
    parameters: dict = field(default_factory=dict)  # JSON Schema style
    requires_auth: bool = False
    category: str = "general"


class ToolRegistry:
    """
    Singleton-style tool registry.
    All tools registered here are available to every agent.
    """
    _instance: Optional["ToolRegistry"] = None

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    @classmethod
    def get(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    # ── Registration ──────────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        description: str,
        fn: Callable,
        parameters: dict | None = None,
        requires_auth: bool = False,
        category: str = "general",
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name, description=description, fn=fn,
            parameters=parameters or {}, requires_auth=requires_auth,
            category=category,
        )
        logger.info("Tool registered: '%s' (%s)", name, category)

    def register_tool(self, name: str, description: str, category: str = "general"):
        """Decorator factory for registering tools."""
        def decorator(fn: Callable) -> Callable:
            self.register(name=name, description=description, fn=fn, category=category)
            return fn
        return decorator

    # ── Invocation ────────────────────────────────────────────────────────────
    def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name, output=None, success=False,
                error=f"Tool '{tool_name}' not found in registry.",
            )
        spec = self._tools[tool_name]
        t0 = time.time()
        try:
            # Support both sync and async functions (sync only here)
            result = spec.fn(**kwargs)
            return ToolResult(
                tool_name=tool_name,
                output=result,
                success=True,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            logger.error("Tool '%s' failed: %s", tool_name, e)
            return ToolResult(
                tool_name=tool_name, output=None, success=False,
                error=str(e), latency_ms=(time.time() - t0) * 1000,
            )

    # ── Discovery ─────────────────────────────────────────────────────────────
    def list_tools(self, category: str | None = None) -> list[dict]:
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        return [
            {"name": t.name, "description": t.description, "category": t.category}
            for t in tools
        ]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # ── Built-in tool stubs ───────────────────────────────────────────────────
    def _register_builtins(self) -> None:
        self.register(
            name="calculator",
            description="Evaluate a mathematical expression safely.",
            fn=self._calculator,
            parameters={"expression": {"type": "string"}},
            category="math",
        )
        self.register(
            name="current_timestamp",
            description="Return the current UTC timestamp.",
            fn=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            category="utility",
        )

    @staticmethod
    def _calculator(expression: str) -> float:
        """Safe expression evaluator (no builtins)."""
        allowed = set("0123456789+-*/().,% ")
        if not all(c in allowed for c in expression):
            raise ValueError(f"Unsafe expression: {expression}")
        return eval(expression, {"__builtins__": {}})  # noqa: S307


# Module-level convenience
registry = ToolRegistry.get()
