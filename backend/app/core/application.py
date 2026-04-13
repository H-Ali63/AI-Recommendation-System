"""
backend/app/core/application.py
────────────────────────────────
FastAPI application factory with full lifecycle management.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from config.logger import configure_logging, get_logger
from config.settings import settings
from backend.app.api import api_router
from backend.app.middleware import LoggingMiddleware, RateLimitMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────
    configure_logging()
    logger.info("Starting RecommendAI API", version=settings.APP_VERSION, env=settings.APP_ENV)

    # Pre-warm the recommendation engine so the first request is fast
    from model.src.engine import get_engine
    engine = get_engine()
    logger.info("Engine pre-warmed", items=len(engine.get_all_titles()))

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down RecommendAI API")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Production-grade cosine-similarity recommendation system.\n\n"
            "Supports content-based, collaborative, and hybrid recommendation modes."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Custom middleware (applied in reverse order) ──────────
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)

    # ── Routes ────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_PREFIX)
    # Root health check without prefix
    from backend.app.api.routes.health import router as health_router
    app.include_router(health_router)

    # ── Global exception handler ──────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception", path=str(request.url), error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc), "status_code": 500},
        )

    return app
