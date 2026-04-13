"""
model/src/engine.py
────────────────────
Singleton factory that wires together preprocessor + recommenders.
All backend services import `get_engine()` — never instantiate directly.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.logger import get_logger
from config.settings import settings
from model.src.preprocessor import DataPreprocessor
from model.src.recommender import (
    CollaborativeRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    RecommendationResult,
)

logger = get_logger(__name__)


class RecommendationEngine:
    """
    High-level facade over the three recommender strategies.

    Usage
    ─────
        engine = RecommendationEngine()
        engine.initialise()
        results = engine.recommend("Inception", top_n=5)
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        self._data_path = data_path or settings.data_path_abs
        self._preprocessor: Optional[DataPreprocessor] = None
        self._content_rec: Optional[ContentBasedRecommender] = None
        self._collab_rec: Optional[CollaborativeRecommender] = None
        self._hybrid_rec: Optional[HybridRecommender] = None
        self._ready: bool = False

    # ── Lifecycle ──────────────────────────────────────────────

    def initialise(
        self,
        interactions: Optional[list[dict]] = None,
        force_recompute: bool = False,
    ) -> None:
        """Load data, build feature matrix, fit all recommenders."""
        logger.info("Initialising recommendation engine …")

        # Preprocessing
        self._preprocessor = DataPreprocessor(self._data_path)
        self._preprocessor.load().preprocess()

        # Content-based
        self._content_rec = ContentBasedRecommender(self._preprocessor)
        self._content_rec.fit(force=force_recompute)

        # Collaborative (optional, requires interaction data)
        self._collab_rec = CollaborativeRecommender(interactions or [])
        self._collab_rec.fit()

        # Hybrid
        self._hybrid_rec = HybridRecommender(self._content_rec, self._collab_rec)

        self._ready = True
        logger.info("Engine ready", items=len(self._preprocessor.titles))

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Recommendation API ─────────────────────────────────────

    def recommend(
        self,
        item_title: str,
        top_n: int = 10,
        category_filter: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: str = "hybrid",  # "content" | "hybrid"
    ) -> list[dict]:
        """
        Main recommendation entrypoint.

        Returns a list of serialisable dicts for the API layer.
        """
        self._assert_ready()

        if mode == "content" or not user_id:
            results = self._content_rec.recommend(
                item_title,
                top_n=top_n,
                category_filter=category_filter,
            )
        else:
            results = self._hybrid_rec.recommend(
                item_title,
                user_id=user_id,
                top_n=top_n,
                category_filter=category_filter,
            )

        return [r.to_dict() for r in results]

    def get_all_titles(self) -> list[str]:
        self._assert_ready()
        return self._preprocessor.titles

    def search_titles(self, query: str, limit: int = 10) -> list[str]:
        self._assert_ready()
        return self._preprocessor.search_titles(query, limit)

    def get_categories(self) -> list[str]:
        self._assert_ready()
        return self._preprocessor.get_categories()

    def get_item_metadata(self, title: str) -> dict:
        self._assert_ready()
        idx = self._preprocessor.get_item_index(title)
        row = self._preprocessor.df.iloc[idx]
        return row.to_dict()

    def get_similarity(self, title_a: str, title_b: str) -> float:
        self._assert_ready()
        return self._content_rec.get_similarity_score(title_a, title_b)

    # ── Helpers ────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("Engine not initialised. Call .initialise() first.")


@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """Return the global singleton engine (initialised on first call)."""
    engine = RecommendationEngine()
    engine.initialise()
    return engine
