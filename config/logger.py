"""
config/logger.py
────────────────
Structured logging setup shared across all services.
Outputs JSON in production, coloured text in development.

Fix: removed structlog.stdlib.add_logger_name from shared_processors —
it calls logger.name which only exists on stdlib Logger objects,
not on structlog's PrintLogger (used by PrintLoggerFactory).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from .settings import settings


def configure_logging() -> None:
    """Call once at application startup."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,          # adds "level" key
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json" or settings.is_production:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so uvicorn/FastAPI access logs flow through
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    configure_logging()
    return structlog.get_logger(name)