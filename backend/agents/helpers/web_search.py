"""DuckDuckGo web search helper for the education agent.

Zero-dependency on cloud keys. If the library isn't available or the call
fails, returns an empty list so the agent can keep working from RAG alone.
"""

from __future__ import annotations

from typing import List


def web_search(query: str, *, max_results: int = 5) -> List[dict]:
    """Return a list of ``{title, url, snippet}`` dicts for ``query``."""
    if not query or not query.strip():
        return []
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError:
        return []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []
    out: List[dict] = []
    for r in raw:
        out.append(
            {
                "title": r.get("title") or "",
                "url": r.get("href") or r.get("url") or "",
                "snippet": r.get("body") or r.get("snippet") or "",
            }
        )
    return out


def web_search_markdown(query: str, *, max_results: int = 5) -> str:
    """Human-readable markdown summary of search results (for LLM consumption)."""
    results = web_search(query, max_results=max_results)
    if not results:
        return f"No web results for: {query}"
    lines = [f"Web results for: {query}", ""]
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. [{r['title']}]({r['url']})")
        if r["snippet"]:
            lines.append(f"   {r['snippet']}")
    return "\n".join(lines)
