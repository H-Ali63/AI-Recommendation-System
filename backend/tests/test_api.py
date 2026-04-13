"""
backend/tests/test_api.py
──────────────────────────
Integration tests for the FastAPI recommendation API.

Run:
    pytest backend/tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Mock the ML engine before importing the app ───────────────
MOCK_TITLES = ["Inception", "The Dark Knight", "Interstellar", "The Prestige"]
MOCK_CATEGORIES = ["Action", "Sci-Fi", "Drama", "Thriller"]
MOCK_RECOMMENDATIONS = [
    {
        "title": "The Dark Knight",
        "similarity_score": 0.85,
        "rank": 1,
        "genres": "Action|Crime|Drama",
        "director": "Christopher Nolan",
        "cast": "Christian Bale|Heath Ledger",
        "rating": 9.0,
        "votes": 2700000,
        "year": 2008,
        "poster_url": "https://example.com/dark_knight.jpg",
    },
    {
        "title": "Interstellar",
        "similarity_score": 0.78,
        "rank": 2,
        "genres": "Adventure|Drama|Sci-Fi",
        "director": "Christopher Nolan",
        "cast": "Matthew McConaughey|Anne Hathaway",
        "rating": 8.6,
        "votes": 1900000,
        "year": 2014,
        "poster_url": "https://example.com/interstellar.jpg",
    },
]


@pytest.fixture(scope="module")
def mock_service():
    """Return a fully mocked RecommendationService."""
    svc = MagicMock()
    svc.engine_stats.return_value = {"ready": True, "items": len(MOCK_TITLES)}
    svc.get_all_titles.return_value = MOCK_TITLES
    svc.get_categories.return_value = MOCK_CATEGORIES
    svc.search_items.return_value = ["Inception", "Interstellar"]
    svc.get_similarity.return_value = 0.8512

    # get_recommendations returns a Pydantic-compatible object
    from backend.app.models.schemas import RecommendedItem, RecommendResponse

    svc.get_recommendations.return_value = RecommendResponse(
        query_item="Inception",
        mode="hybrid",
        total=2,
        recommendations=[RecommendedItem(**r) for r in MOCK_RECOMMENDATIONS],
        cached=False,
    )
    svc.get_item_metadata.return_value = {
        "title": "Inception",
        "genres": "Action|Adventure|Sci-Fi",
        "director": "Christopher Nolan",
        "cast": "Leonardo DiCaprio|Joseph Gordon-Levitt",
        "rating": 8.8,
        "votes": 2300000,
        "year": 2010,
        "poster_url": "https://example.com/inception.jpg",
    }
    return svc


@pytest.fixture(scope="module")
def client(mock_service):
    """FastAPI test client with mocked service."""
    with patch(
        "backend.app.services.recommendation_service.get_recommendation_service",
        return_value=mock_service,
    ):
        with patch(
            "backend.app.api.routes.recommend.get_recommendation_service",
            return_value=mock_service,
        ):
            with patch(
                "backend.app.api.routes.health.get_recommendation_service",
                return_value=mock_service,
            ):
                with patch("model.src.engine.get_engine"):
                    from backend.app.core.application import create_app
                    app = create_app()
                    yield TestClient(app, raise_server_exceptions=True)


# ════════════════════════════════════════════════════════════════
#  Health endpoint
# ════════════════════════════════════════════════════════════════
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_structure(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "engine_ready" in data
        assert "items_loaded" in data
        assert data["engine_ready"] is True
        assert data["items_loaded"] == len(MOCK_TITLES)

    def test_root_returns_service_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert "service" in data
        assert "version" in data


# ════════════════════════════════════════════════════════════════
#  Recommendation endpoint
# ════════════════════════════════════════════════════════════════
class TestRecommendations:
    def test_recommend_returns_200(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5},
        )
        assert r.status_code == 200

    def test_recommend_response_structure(self, client):
        data = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5},
        ).json()
        assert "query_item" in data
        assert "recommendations" in data
        assert "total" in data
        assert "mode" in data
        assert isinstance(data["recommendations"], list)

    def test_recommend_items_have_required_fields(self, client):
        recs = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5},
        ).json()["recommendations"]
        for item in recs:
            assert "title" in item
            assert "similarity_score" in item
            assert "rank" in item
            assert 0.0 <= item["similarity_score"] <= 1.0

    def test_recommend_with_category_filter(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5, "category_filter": "Sci-Fi"},
        )
        assert r.status_code == 200

    def test_recommend_with_hybrid_mode(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5, "mode": "hybrid", "user_id": "user_42"},
        )
        assert r.status_code == 200

    def test_recommend_invalid_mode_rejected(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 5, "mode": "invalid"},
        )
        assert r.status_code == 422

    def test_recommend_top_n_too_large_rejected(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "Inception", "top_n": 999},
        )
        assert r.status_code == 422

    def test_recommend_empty_title_rejected(self, client):
        r = client.post(
            "/api/v1/recommend/",
            json={"item_title": "", "top_n": 5},
        )
        assert r.status_code == 422


# ════════════════════════════════════════════════════════════════
#  Search & Catalog
# ════════════════════════════════════════════════════════════════
class TestCatalog:
    def test_get_all_titles(self, client):
        r = client.get("/api/v1/recommend/titles")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == len(MOCK_TITLES)

    def test_search_titles(self, client):
        r = client.get("/api/v1/recommend/search?q=inception")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_search_missing_query_rejected(self, client):
        r = client.get("/api/v1/recommend/search")
        assert r.status_code == 422

    def test_get_categories(self, client):
        r = client.get("/api/v1/recommend/categories")
        assert r.status_code == 200
        data = r.json()
        assert "categories" in data
        assert "total" in data
        assert isinstance(data["categories"], list)

    def test_get_item_metadata(self, client):
        r = client.get("/api/v1/recommend/item/Inception")
        assert r.status_code == 200
        data = r.json()
        assert data["title"] == "Inception"
        assert "genres" in data
        assert "rating" in data


# ════════════════════════════════════════════════════════════════
#  Similarity
# ════════════════════════════════════════════════════════════════
class TestSimilarity:
    def test_similarity_score(self, client):
        r = client.post(
            "/api/v1/recommend/similarity",
            json={"title_a": "Inception", "title_b": "Interstellar"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "similarity_score" in data
        assert 0.0 <= data["similarity_score"] <= 1.0
        assert data["title_a"] == "Inception"
        assert data["title_b"] == "Interstellar"
