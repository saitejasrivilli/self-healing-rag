"""
tools/api_tools.py
==================
HTTP-based tool implementations: REST calls, web search, webhooks.
All tools self-register into ToolRegistry on import.
"""

from __future__ import annotations
import json
import logging
import os
import urllib.request
import urllib.parse
from typing import Any

from tools.tool_registry import registry

logger = logging.getLogger(__name__)


# ── HTTP GET ───────────────────────────────────────────────────────────────────
@registry.register_tool(
    name="http_get",
    description="Perform an HTTP GET request and return the response body.",
    category="api",
)
def http_get(url: str, headers: dict | None = None, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return {
                "status": resp.status,
                "body": body[:4000],   # cap response size
                "content_type": resp.headers.get("Content-Type", ""),
            }
    except Exception as e:
        return {"status": -1, "body": "", "error": str(e)}


# ── Web search (stub — swap for SerpAPI / Tavily) ─────────────────────────────
@registry.register_tool(
    name="web_search",
    description="Search the web for current information on a topic.",
    category="api",
)
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Stub implementation.
    Replace with:
      - SerpAPI: https://serpapi.com
      - Tavily: https://tavily.com
      - DuckDuckGo: duckduckgo_search library
    """
    logger.info("web_search called: '%s' (stub — returning placeholder)", query)
    return [
        {
            "title": f"[Stub] Result {i+1} for: {query}",
            "url": f"https://example.com/result-{i+1}",
            "snippet": f"This is a placeholder result. Configure a real search API via SEARCH_API_KEY.",
        }
        for i in range(min(num_results, 3))
    ]


# ── JSON API fetch ─────────────────────────────────────────────────────────────
@registry.register_tool(
    name="fetch_json",
    description="Fetch JSON data from a REST API endpoint.",
    category="api",
)
def fetch_json(url: str, params: dict | None = None, timeout: int = 10) -> Any:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    result = http_get(url, timeout=timeout)
    try:
        return json.loads(result["body"])
    except json.JSONDecodeError:
        return result


# ── Webhook sender ─────────────────────────────────────────────────────────────
@registry.register_tool(
    name="send_webhook",
    description="POST a JSON payload to a webhook URL.",
    category="api",
)
def send_webhook(url: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": resp.status, "ok": resp.status < 400}
    except Exception as e:
        return {"status": -1, "ok": False, "error": str(e)}
