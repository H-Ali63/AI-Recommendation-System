"""
config/settings.py
──────────────────
Centralised, environment-driven configuration for RecommendAI.
All services import from here — never hard-code values elsewhere.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Resolve project root (two levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class AppSettings(BaseSettings):
    # ── Application ────────────────────────────────────────────
    APP_ENV: str = "development"
    APP_NAME: str = "RecommendAI"
    APP_VERSION: str = "1.0.0"

    # ── Backend ────────────────────────────────────────────────
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: str = "http://localhost:8501"

    # ── Frontend ───────────────────────────────────────────────
    FRONTEND_PORT: int = 8501
    BACKEND_URL: str = "http://localhost:8000"

    # ── MongoDB ────────────────────────────────────────────────
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "recommendai"
    MONGO_COLLECTION_ITEMS: str = "movies"
    MONGO_COLLECTION_INTERACTIONS: str = "interactions"
    MONGO_COLLECTION_CACHE: str = "recommendation_cache"

    # ── Redis ──────────────────────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 3600

    # ── ML Model ───────────────────────────────────────────────
    DATA_PATH: str = "data/raw/movies.csv"
    MODEL_CACHE_DIR: str = "model/cache"
    TOP_N_DEFAULT: int = 10
    SIMILARITY_THRESHOLD: float = 0.1

    # ── Logging ────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ── Rate Limiting ──────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    class Config:
        env_file = str(ROOT_DIR / ".env")
        case_sensitive = True

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def data_path_abs(self) -> Path:
        return ROOT_DIR / self.DATA_PATH

    @property
    def model_cache_path(self) -> Path:
        p = ROOT_DIR / self.MODEL_CACHE_DIR
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached singleton settings object."""
    return AppSettings()


settings = get_settings()
