"""
backend/app/services/recommendation_service.py
───────────────────────────────────────────────
Business logic layer between API routes and the ML engine.
Handles caching, error translation, and response shaping.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from config.logger import get_logger
from model.src.engine import get_engine
from backend.app.models.schemas import RecommendedItem, RecommendResponse
from backend.app.services.cache_service import get_cache

logger = get_logger(__name__)


class RecommendationService:

    def __init__(self) -> None:
        self._engine = get_engine()
        self._cache = get_cache()

    def get_recommendations(
        self,
        item_title: str,
        top_n: int = 10,
        category_filter: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: str = "hybrid",
    ) -> RecommendResponse:
        """
        Fetch recommendations, using cache when available.
        """
        cache_key = self._cache.make_key(
            "rec", item_title, top_n, category_filter, user_id, mode
        )

        # Cache hit
        cached = self._cache.get(cache_key)
        if cached:
            logger.info("Cache hit", key=cache_key)
            response = RecommendResponse(**cached)
            response.cached = True
            return response

        # Engine call
        logger.info(
            "Computing recommendations",
            item=item_title,
            top_n=top_n,
            mode=mode,
        )
        raw_results = self._engine.recommend(
            item_title=item_title,
            top_n=top_n,
            category_filter=category_filter,
            user_id=user_id,
            mode=mode,
        )

        items = [RecommendedItem(**r) for r in raw_results]
        response = RecommendResponse(
            query_item=item_title,
            mode=mode,
            total=len(items),
            recommendations=items,
            cached=False,
        )

        # Store in cache
        self._cache.set(cache_key, response.model_dump())
        return response

    def search_items(self, query: str, limit: int = 10) -> list[str]:
        return self._engine.search_titles(query, limit)

    def get_all_titles(self) -> list[str]:
        cache_key = "recsys:all_titles"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        titles = self._engine.get_all_titles()
        self._cache.set(cache_key, titles, ttl=86400)  # 24h TTL
        return titles

    def get_categories(self) -> list[str]:
        cache_key = "recsys:categories"
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        cats = self._engine.get_categories()
        self._cache.set(cache_key, cats, ttl=86400)
        return cats

    def get_item_metadata(self, title: str) -> dict:
        cache_key = self._cache.make_key("meta", title)
        cached = self._cache.get(cache_key)
        if cached:
            return cached
        meta = self._engine.get_item_metadata(title)
        self._cache.set(cache_key, meta, ttl=86400)
        return meta

    def get_similarity(self, title_a: str, title_b: str) -> float:
        return self._engine.get_similarity(title_a, title_b)

    def engine_stats(self) -> dict:
        return {
            "ready": self._engine.is_ready,
            "items": len(self._engine.get_all_titles()) if self._engine.is_ready else 0,
        }


# Singleton
_service: Optional[RecommendationService] = None


def get_recommendation_service() -> RecommendationService:
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service
