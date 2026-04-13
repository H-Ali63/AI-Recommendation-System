"""
backend/app/api/routes/health.py
─────────────────────────────────
Health, readiness, and version endpoints.
Used by Docker health-checks, load balancers, and Kubernetes probes.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from backend.app.models.schemas import HealthResponse
from backend.app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)
from config.settings import settings

router = APIRouter(tags=["Health"])

_START_TIME = time.time()


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(
    svc: RecommendationService = Depends(get_recommendation_service),
) -> HealthResponse:
    stats = svc.engine_stats()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        engine_ready=stats["ready"],
        items_loaded=stats["items"],
    )


@router.get("/", summary="Root — service info")
async def root() -> dict:
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "uptime_seconds": round(time.time() - _START_TIME, 2),
        "docs": "/docs",
    }
