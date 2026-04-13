"""
frontend/api_client.py
───────────────────────
HTTP client for the RecommendAI backend API.
All network calls go through this module — never call requests directly from pages.
"""
from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Backend URL from environment (Docker sets this; default for local dev)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = "/api/v1"
BASE_URL = f"{BACKEND_URL}{API_PREFIX}"

_TIMEOUT = 30  # seconds


def _session() -> requests.Session:
    """Return a session with retry logic."""
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


# ── API calls ────────────────────────────────────────────────────

def health_check() -> dict:
    try:
        r = _session().get(f"{BACKEND_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"status": "error", "detail": str(exc), "engine_ready": False, "items_loaded": 0}


def get_recommendations(
    item_title: str,
    top_n: int = 10,
    category_filter: Optional[str] = None,
    user_id: Optional[str] = None,
    mode: str = "hybrid",
) -> dict:
    """POST /recommend/ — returns full RecommendResponse dict."""
    payload = {
        "item_title": item_title,
        "top_n": top_n,
        "mode": mode,
    }
    if category_filter:
        payload["category_filter"] = category_filter
    if user_id:
        payload["user_id"] = user_id

    r = _session().post(f"{BASE_URL}/recommend/", json=payload, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def search_titles(query: str, limit: int = 20) -> list[str]:
    r = _session().get(
        f"{BASE_URL}/recommend/search",
        params={"q": query, "limit": limit},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_all_titles() -> list[str]:
    r = _session().get(f"{BASE_URL}/recommend/titles", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_categories() -> list[str]:
    r = _session().get(f"{BASE_URL}/recommend/categories", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json().get("categories", [])


def get_item_metadata(title: str) -> dict:
    r = _session().get(
        f"{BASE_URL}/recommend/item/{requests.utils.quote(title)}",
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def get_similarity(title_a: str, title_b: str) -> float:
    r = _session().post(
        f"{BASE_URL}/recommend/similarity",
        json={"title_a": title_a, "title_b": title_b},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json().get("similarity_score", 0.0)
