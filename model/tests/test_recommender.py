"""
model/tests/test_recommender.py
────────────────────────────────
Unit tests for the ML recommendation engine.

Run:
    pytest model/tests/ -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.src.preprocessor import DataPreprocessor
from model.src.recommender import (
    CollaborativeRecommender,
    ContentBasedRecommender,
    HybridRecommender,
    RecommendationResult,
)


# ── Fixtures ──────────────────────────────────────────────────

SAMPLE_CSV = """movie_id,title,genres,director,cast,rating,votes,year,poster_url
1,Movie Alpha,Action|Drama,Director A,Actor X|Actor Y,8.5,100000,2010,
2,Movie Beta,Action|Comedy,Director B,Actor X|Actor Z,7.8,80000,2012,
3,Movie Gamma,Drama|Thriller,Director A,Actor Y|Actor W,8.1,90000,2015,
4,Movie Delta,Comedy,Director C,Actor Z|Actor V,7.2,60000,2018,
5,Movie Epsilon,Action|Sci-Fi,Director B,Actor X|Actor W,8.9,150000,2020,
"""


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "test_movies.csv"
    path.write_text(SAMPLE_CSV)
    return path


@pytest.fixture(scope="module")
def preprocessor(sample_csv):
    pp = DataPreprocessor(sample_csv)
    pp.load().preprocess()
    return pp


@pytest.fixture(scope="module")
def content_rec(preprocessor):
    rec = ContentBasedRecommender(preprocessor)
    rec.fit()
    return rec


# ════════════════════════════════════════════════════════════════
#  DataPreprocessor Tests
# ════════════════════════════════════════════════════════════════
class TestDataPreprocessor:
    def test_load_creates_dataframe(self, preprocessor):
        assert preprocessor.df is not None
        assert len(preprocessor.df) == 5

    def test_titles_list(self, preprocessor):
        titles = preprocessor.titles
        assert len(titles) == 5
        assert "Movie Alpha" in titles

    def test_feature_matrix_shape(self, preprocessor):
        matrix = preprocessor.feature_matrix
        assert matrix is not None
        assert matrix.shape[0] == 5
        assert matrix.shape[1] > 0

    def test_feature_matrix_is_numeric(self, preprocessor):
        assert preprocessor.feature_matrix.dtype in [np.float32, np.float64]
        assert not np.isnan(preprocessor.feature_matrix).any()

    def test_get_item_index_exact_match(self, preprocessor):
        idx = preprocessor.get_item_index("Movie Alpha")
        assert idx == 0

    def test_get_item_index_case_insensitive(self, preprocessor):
        idx = preprocessor.get_item_index("movie alpha")
        assert idx == 0

    def test_get_item_index_not_found_raises(self, preprocessor):
        with pytest.raises(ValueError, match="not found"):
            preprocessor.get_item_index("Does Not Exist")

    def test_search_titles_returns_matches(self, preprocessor):
        results = preprocessor.search_titles("movie")
        assert len(results) == 5  # all match "movie"

    def test_search_titles_case_insensitive(self, preprocessor):
        results = preprocessor.search_titles("ALPHA")
        assert "Movie Alpha" in results

    def test_get_categories(self, preprocessor):
        cats = preprocessor.get_categories()
        assert "Action" in cats
        assert "Drama" in cats
        assert isinstance(cats, list)
        assert cats == sorted(cats)

    def test_missing_file_raises(self):
        pp = DataPreprocessor("/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            pp.load()


# ════════════════════════════════════════════════════════════════
#  ContentBasedRecommender Tests
# ════════════════════════════════════════════════════════════════
class TestContentBasedRecommender:
    def test_fit_builds_similarity_matrix(self, content_rec, preprocessor):
        n = len(preprocessor.titles)
        assert content_rec._sim_matrix is not None
        assert content_rec._sim_matrix.shape == (n, n)

    def test_similarity_matrix_diagonal_is_one(self, content_rec, preprocessor):
        n = len(preprocessor.titles)
        diag = np.diag(content_rec._sim_matrix)
        np.testing.assert_allclose(diag, np.ones(n), atol=1e-5)

    def test_similarity_matrix_symmetric(self, content_rec):
        m = content_rec._sim_matrix
        np.testing.assert_allclose(m, m.T, atol=1e-5)

    def test_similarity_matrix_values_in_range(self, content_rec):
        m = content_rec._sim_matrix
        assert m.min() >= -0.01  # floating point tolerance
        assert m.max() <= 1.01

    def test_recommend_returns_list(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=3)
        assert isinstance(results, list)

    def test_recommend_excludes_self(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=10)
        titles = [r.title for r in results]
        assert "Movie Alpha" not in titles

    def test_recommend_respects_top_n(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=2)
        assert len(results) <= 2

    def test_recommend_sorted_descending(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_rank_sequential(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=3)
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    def test_recommend_category_filter(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=5, category_filter="Comedy")
        for r in results:
            assert "Comedy" in r.metadata.get("genres", "")

    def test_recommend_result_has_metadata(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=2)
        for r in results:
            assert isinstance(r.metadata, dict)
            assert "genres" in r.metadata

    def test_recommend_unknown_title_raises(self, content_rec):
        with pytest.raises(ValueError):
            content_rec.recommend("This Movie Does Not Exist", top_n=5)

    def test_get_similarity_score_self(self, content_rec):
        score = content_rec.get_similarity_score("Movie Alpha", "Movie Alpha")
        assert pytest.approx(score, abs=1e-4) == 1.0

    def test_get_similarity_score_range(self, content_rec):
        score = content_rec.get_similarity_score("Movie Alpha", "Movie Beta")
        assert 0.0 <= score <= 1.0

    def test_to_dict_serialisable(self, content_rec):
        results = content_rec.recommend("Movie Alpha", top_n=1)
        import json
        d = results[0].to_dict()
        json.dumps(d)  # Should not raise


# ════════════════════════════════════════════════════════════════
#  CollaborativeRecommender Tests
# ════════════════════════════════════════════════════════════════
class TestCollaborativeRecommender:
    @pytest.fixture
    def interactions(self):
        return [
            {"user_id": "u1", "item_title": "Movie Alpha", "rating": 5},
            {"user_id": "u1", "item_title": "Movie Beta",  "rating": 4},
            {"user_id": "u2", "item_title": "Movie Alpha", "rating": 4},
            {"user_id": "u2", "item_title": "Movie Gamma", "rating": 5},
            {"user_id": "u3", "item_title": "Movie Beta",  "rating": 3},
            {"user_id": "u3", "item_title": "Movie Delta", "rating": 4},
        ]

    def test_fit_with_interactions(self, interactions):
        rec = CollaborativeRecommender(interactions)
        rec.fit()
        assert rec._matrix is not None
        assert len(rec._users) == 3

    def test_fit_empty_interactions(self):
        rec = CollaborativeRecommender([])
        rec.fit()
        assert rec._matrix is None

    def test_recommend_known_user(self, interactions):
        rec = CollaborativeRecommender(interactions)
        rec.fit()
        results = rec.recommend("u1", top_n=3)
        assert isinstance(results, list)
        # Should not recommend already-rated items
        assert "Movie Alpha" not in results
        assert "Movie Beta" not in results

    def test_recommend_unknown_user_returns_empty(self, interactions):
        rec = CollaborativeRecommender(interactions)
        rec.fit()
        results = rec.recommend("unknown_user", top_n=5)
        assert results == []


# ════════════════════════════════════════════════════════════════
#  HybridRecommender Tests
# ════════════════════════════════════════════════════════════════
class TestHybridRecommender:
    @pytest.fixture
    def hybrid(self, content_rec):
        interactions = [
            {"user_id": "u1", "item_title": "Movie Alpha", "rating": 5},
            {"user_id": "u1", "item_title": "Movie Gamma", "rating": 4},
        ]
        collab = CollaborativeRecommender(interactions)
        collab.fit()
        return HybridRecommender(content_rec, collab, alpha=0.7)

    def test_hybrid_without_user_falls_back_to_content(self, hybrid):
        results = hybrid.recommend("Movie Alpha", user_id=None, top_n=3)
        assert len(results) > 0

    def test_hybrid_with_user_returns_results(self, hybrid):
        results = hybrid.recommend("Movie Alpha", user_id="u1", top_n=3)
        assert len(results) > 0

    def test_hybrid_scores_in_range(self, hybrid):
        results = hybrid.recommend("Movie Alpha", user_id="u1", top_n=3)
        for r in results:
            assert 0.0 <= r.score <= 1.5  # hybrid can slightly exceed 1.0

    def test_hybrid_respects_top_n(self, hybrid):
        results = hybrid.recommend("Movie Alpha", user_id="u1", top_n=2)
        assert len(results) <= 2


# ════════════════════════════════════════════════════════════════
#  RecommendationResult Tests
# ════════════════════════════════════════════════════════════════
class TestRecommendationResult:
    def test_to_dict_contains_required_keys(self):
        r = RecommendationResult(
            title="Test Movie",
            score=0.75,
            rank=1,
            metadata={"genres": "Action", "rating": 8.5},
        )
        d = r.to_dict()
        assert d["title"] == "Test Movie"
        assert d["similarity_score"] == 0.75
        assert d["rank"] == 1
        assert d["genres"] == "Action"

    def test_score_rounded_to_4_decimals(self):
        r = RecommendationResult(title="X", score=0.123456789, rank=1)
        d = r.to_dict()
        assert d["similarity_score"] == round(0.123456789, 4)
