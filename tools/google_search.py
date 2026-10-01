"""Live web search tool. Prefers Google Programmable Search, then DuckDuckGo."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import tool

_MAX_RESULTS = 5


def _format_results(rows: list[dict[str, str]], source: str) -> str:
    if not rows:
        return f"No results from {source}."
    blocks = []
    for i, row in enumerate(rows, start=1):
        title = row.get("title") or "(no title)"
        snippet = row.get("snippet") or ""
        link = row.get("link") or ""
        blocks.append(f"{i}. {title}\n   {snippet}\n   Source: {link}")
    return f"Search backend: {source}\n" + "\n".join(blocks)


def _google_cse_search(query: str) -> str | None:
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        return None
    try:
        from googleapiclient.discovery import build

        service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
        response: dict[str, Any] = (
            service.cse().list(q=query, cx=cse_id, num=_MAX_RESULTS).execute()
        )
        items = response.get("items") or []
        rows = [
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", ""),
            }
            for item in items
        ]
        return _format_results(rows, "Google Custom Search")
    except Exception as exc:  # noqa: BLE001
        return f"Google Search failed ({exc}). Falling back if possible."


def _ddg_search(query: str) -> str:
    try:
        from langchain_community.tools import DuckDuckGoSearchRun

        raw = DuckDuckGoSearchRun().run(query)
        if raw:
            return f"Search backend: DuckDuckGoSearchRun\n{raw}"
    except Exception:
        pass

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
        except ImportError as exc:
            return (
                "DuckDuckGo search is unavailable. Install `ddgs` "
                f"or set GOOGLE_API_KEY and GOOGLE_CSE_ID. ({exc})"
            )

    rows: list[dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=_MAX_RESULTS):
                rows.append(
                    {
                        "title": hit.get("title") or "",
                        "snippet": hit.get("body") or hit.get("snippet") or "",
                        "link": hit.get("href") or hit.get("link") or "",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        return f"DuckDuckGo search failed: {exc}"
    return _format_results(rows, "DuckDuckGo")


def run_web_search(query: str) -> str:
    google = _google_cse_search(query)
    if google and not google.startswith("Google Search failed"):
        return google
    ddg = _ddg_search(query)
    if google and google.startswith("Google Search failed"):
        return f"{google}\n{ddg}"
    return ddg


@tool
def google_search(query: str) -> str:
    """Search the live web for current facts, prices, places, news, and sources.

    Use this when you need up-to-date information that is not a pure calculation.
    Returns titles, snippets, and source URLs.
    """
    return run_web_search(query)
