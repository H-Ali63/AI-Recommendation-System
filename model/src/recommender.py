"""
model/src/recommender.py
─────────────────────────
Core recommendation engine.

Supports:
  • Content-based filtering   → cosine similarity on feature matrix
  • Collaborative filtering   → user-item interaction matrix (basic)
  • Hybrid                    → weighted blend of both scores
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.logger import get_logger
from config.settings import settings
from model.src.preprocessor import DataPreprocessor

logger = get_logger(__name__)


@dataclass
class RecommendationResult:
    title: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "similarity_score": round(self.score, 4),
            "rank": self.rank,
            **self.metadata,
        }


class ContentBasedRecommender:
    """
    Item-based content recommender using cosine similarity.

    Architecture
    ────────────
    fit()  → builds N×N similarity matrix (cached to disk)
    recommend() → looks up pre-computed row, applies filters, returns top-N
    """

    _CACHE_FILE = "similarity_matrix.pkl"
    _META_FILE = "item_metadata.pkl"

    def __init__(self, preprocessor: DataPreprocessor) -> None:
        self.preprocessor = preprocessor
        self._sim_matrix: np.ndarray | None = None
        self._cache_path = settings.model_cache_path / self._CACHE_FILE
        self._meta_path = settings.model_cache_path / self._META_FILE

    # ── Lifecycle ──────────────────────────────────────────────

    def fit(self, force: bool = False) -> ContentBasedRecommender:
        """Build or load the cosine similarity matrix."""
        if not force and self._try_load_cache():
            return self

        logger.info("Computing cosine similarity matrix …")
        matrix = self.preprocessor.feature_matrix
        self._sim_matrix = cosine_similarity(matrix, matrix)
        logger.info(
            "Similarity matrix built",
            shape=self._sim_matrix.shape,
            memory_mb=round(self._sim_matrix.nbytes / 1024 / 1024, 2),
        )
        self._save_cache()
        return self

    # ── Public API ─────────────────────────────────────────────

    def recommend(
        self,
        item_title: str,
        top_n: int = 10,
        category_filter: Optional[str] = None,
        min_score: float = 0.0,
        exclude_self: bool = True,
    ) -> list[RecommendationResult]:
        """
        Return top-N recommendations for a given item title.

        Parameters
        ──────────
        item_title     : exact or case-insensitive title match
        top_n          : number of results to return
        category_filter: filter results to a specific genre
        min_score      : minimum cosine similarity threshold
        exclude_self   : whether to exclude the query item itself
        """
        self._ensure_fit()

        idx = self.preprocessor.get_item_index(item_title)
        scores = self._sim_matrix[idx].copy()

        df = self.preprocessor.df.copy()
        df["_score"] = scores

        if exclude_self:
            df = df[df.index != idx]

        # Apply similarity threshold
        df = df[df["_score"] >= max(min_score, settings.SIMILARITY_THRESHOLD)]

        # Apply category filter
        if category_filter:
            df = df[df["genres"].str.contains(category_filter, case=False, na=False)]

        # Sort and paginate
        df = df.sort_values("_score", ascending=False).head(top_n)

        results = []
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            metadata = self._extract_metadata(row)
            results.append(
                RecommendationResult(
                    title=row["title"],
                    score=float(row["_score"]),
                    rank=rank,
                    metadata=metadata,
                )
            )

        logger.info(
            "Recommendations generated",
            query=item_title,
            results=len(results),
            top_n=top_n,
        )
        return results

    def recommend_by_index(self, idx: int, top_n: int = 10) -> list[RecommendationResult]:
        """Internal helper: recommend by DataFrame row index."""
        title = self.preprocessor.df.iloc[idx]["title"]
        return self.recommend(title, top_n=top_n)

    def get_similarity_score(self, title_a: str, title_b: str) -> float:
        """Return the pairwise cosine similarity between two items."""
        self._ensure_fit()
        idx_a = self.preprocessor.get_item_index(title_a)
        idx_b = self.preprocessor.get_item_index(title_b)
        return float(self._sim_matrix[idx_a, idx_b])

    # ── Cache management ────────────────────────────────────────

    def _save_cache(self) -> None:
        with open(self._cache_path, "wb") as f:
            pickle.dump(self._sim_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Similarity matrix cached", path=str(self._cache_path))

    def _try_load_cache(self) -> bool:
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "rb") as f:
                    self._sim_matrix = pickle.load(f)
                logger.info("Loaded similarity matrix from cache")
                return True
            except Exception as exc:
                logger.warning("Cache load failed, will recompute", error=str(exc))
        return False

    def _ensure_fit(self) -> None:
        if self._sim_matrix is None:
            raise RuntimeError("Recommender not fitted. Call .fit() first.")

    @staticmethod
    def _extract_metadata(row: pd.Series) -> dict:
        return {
            k: row[k]
            for k in ["genres", "director", "cast", "rating", "votes", "year", "poster_url"]
            if k in row.index
        }


class CollaborativeRecommender:
    """
    Lightweight user-based collaborative filtering.
    Uses a user-item rating matrix; computes user similarity
    and predicts unseen item scores for the target user.

    Note: In production, replace with ALS or SVD via Surprise/implicit.
    """

    def __init__(self, interactions: list[dict]) -> None:
        """
        interactions: list of {user_id, item_title, rating} dicts
        """
        self._interactions = interactions
        self._matrix: pd.DataFrame | None = None
        self._user_sim: np.ndarray | None = None
        self._users: list[str] = []

    def fit(self) -> CollaborativeRecommender:
        if not self._interactions:
            logger.warning("No interaction data — collaborative filtering disabled")
            return self

        df = pd.DataFrame(self._interactions)
        self._matrix = df.pivot_table(
            index="user_id", columns="item_title", values="rating", fill_value=0
        )
        self._users = list(self._matrix.index)
        user_matrix = self._matrix.values
        self._user_sim = cosine_similarity(user_matrix, user_matrix)
        logger.info("Collaborative model fitted", users=len(self._users))
        return self

    def recommend(self, user_id: str, top_n: int = 10) -> list[str]:
        """Return top-N item titles predicted for a user."""
        if self._matrix is None or user_id not in self._users:
            return []

        u_idx = self._users.index(user_id)
        sim_scores = self._user_sim[u_idx]

        # Weighted average of other users' ratings
        rated = self._matrix.values
        sim_col = sim_scores[:, np.newaxis]
        predicted = (sim_col * rated).sum(axis=0) / (sim_scores.sum() + 1e-9)

        # Exclude already-rated items
        user_rated_mask = (self._matrix.iloc[u_idx] > 0).values
        predicted[user_rated_mask] = -1

        top_idxs = np.argsort(predicted)[::-1][:top_n]
        return [self._matrix.columns[i] for i in top_idxs if predicted[i] > 0]


class HybridRecommender:
    """
    Blends content-based and collaborative signals.

    Score_hybrid = α × score_content + (1-α) × score_collab
    """

    def __init__(
        self,
        content_rec: ContentBasedRecommender,
        collab_rec: CollaborativeRecommender,
        alpha: float = 0.7,
    ) -> None:
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.alpha = alpha  # weight for content-based

    def recommend(
        self,
        item_title: str,
        user_id: Optional[str] = None,
        top_n: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[RecommendationResult]:
        """
        If user_id is provided, mix collaborative signal.
        Otherwise fall back to pure content-based.
        """
        content_results = self.content_rec.recommend(
            item_title, top_n=top_n * 2, category_filter=category_filter
        )

        if user_id is None:
            return content_results[:top_n]

        collab_titles = set(self.collab_rec.recommend(user_id, top_n=top_n * 2))

        boosted: list[RecommendationResult] = []
        for r in content_results:
            collab_score = 1.0 if r.title in collab_titles else 0.0
            hybrid_score = self.alpha * r.score + (1 - self.alpha) * collab_score
            boosted.append(
                RecommendationResult(
                    title=r.title,
                    score=hybrid_score,
                    rank=r.rank,
                    metadata=r.metadata,
                )
            )

        boosted.sort(key=lambda x: x.score, reverse=True)
        for i, r in enumerate(boosted[:top_n], start=1):
            r.rank = i
        return boosted[:top_n]
