"""
backend/app/models/schemas.py
──────────────────────────────
Pydantic v2 schemas for all API request / response payloads.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schemas ─────────────────────────────────────────────

class RecommendRequest(BaseModel):
    item_title: str = Field(..., min_length=1, max_length=200, description="Item to base recommendations on")
    top_n: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    category_filter: Optional[str] = Field(default=None, description="Filter by genre/category")
    user_id: Optional[str] = Field(default=None, description="User ID for hybrid recommendations")
    mode: str = Field(default="hybrid", description="Recommendation mode: content | hybrid")

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"content", "hybrid"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "item_title": "Inception",
            "top_n": 5,
            "category_filter": "Sci-Fi",
            "mode": "hybrid",
        }
    }}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=50)


class SimilarityRequest(BaseModel):
    title_a: str = Field(..., min_length=1)
    title_b: str = Field(..., min_length=1)


# ── Response schemas ────────────────────────────────────────────

class RecommendedItem(BaseModel):
    title: str
    similarity_score: float
    rank: int
    genres: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    rating: Optional[float] = None
    votes: Optional[int] = None
    year: Optional[int] = None
    poster_url: Optional[str] = None


class RecommendResponse(BaseModel):
    query_item: str
    mode: str
    total: int
    recommendations: list[RecommendedItem]
    cached: bool = False


class ItemMetadataResponse(BaseModel):
    title: str
    genres: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    rating: Optional[float] = None
    votes: Optional[int] = None
    year: Optional[int] = None
    poster_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    engine_ready: bool
    items_loaded: int


class CategoriesResponse(BaseModel):
    categories: list[str]
    total: int


class SimilarityResponse(BaseModel):
    title_a: str
    title_b: str
    similarity_score: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
