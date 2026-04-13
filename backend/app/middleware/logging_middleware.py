"""
backend/app/middleware/logging_middleware.py
─────────────────────────────────────────────
Request/response logging middleware with timing and rate limiting.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            client=request.client.host if request.client else "unknown",
        )
        response.headers["X-Process-Time-Ms"] = str(duration_ms)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter (per client IP).
    Settings: RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds.
    """

    def __init__(self, app) -> None:
        super().__init__(app)
        self._windows: dict[str, deque] = defaultdict(deque)
        self._limit = settings.RATE_LIMIT_REQUESTS
        self._window = settings.RATE_LIMIT_WINDOW

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting on health checks
        if request.url.path in {"/health", "/"}:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self._windows[client_ip]

        # Evict timestamps outside the current window
        while window and window[0] < now - self._window:
            window.popleft()

        if len(window) >= self._limit:
            logger.warning("Rate limit exceeded", client=client_ip)
            return Response(
                content='{"error":"Rate limit exceeded","status_code":429}',
                status_code=429,
                media_type="application/json",
                headers={
                    "Retry-After": str(self._window),
                    "X-RateLimit-Limit": str(self._limit),
                },
            )

        window.append(now)
        return await call_next(request)
