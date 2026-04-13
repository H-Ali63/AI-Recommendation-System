"""
backend/app/services/cache_service.py
──────────────────────────────────────
Redis-backed cache with automatic in-memory fallback.
All recommendation results are stored as JSON strings keyed on a
deterministic hash of the request parameters.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CacheService:
    """
    Two-tier cache:
      1. Redis (preferred, distributed)
      2. In-process dict (fallback if Redis unavailable)
    """

    def __init__(self) -> None:
        self._redis = None
        self._local: dict[str, str] = {}
        self._ttl = settings.REDIS_TTL
        self._connect_redis()

    def _connect_redis(self) -> None:
        try:
            import redis

            client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                socket_connect_timeout=2,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            logger.info("Redis cache connected", host=settings.REDIS_HOST)
        except Exception as exc:
            logger.warning("Redis unavailable — using in-memory cache", reason=str(exc))
            self._redis = None

    # ── Public interface ────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return deserialised value or None on miss."""
        raw = self._raw_get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Serialise and store value. Returns True on success."""
        try:
            raw = json.dumps(value, default=str)
            return self._raw_set(key, raw, ttl or self._ttl)
        except (TypeError, ValueError) as exc:
            logger.error("Cache serialisation failed", error=str(exc))
            return False

    def delete(self, key: str) -> bool:
        if self._redis:
            return bool(self._redis.delete(key))
        self._local.pop(key, None)
        return True

    def flush(self) -> bool:
        """Clear all cached entries (use carefully)."""
        if self._redis:
            self._redis.flushdb()
        self._local.clear()
        logger.info("Cache flushed")
        return True

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Build a deterministic cache key from arbitrary parts."""
        raw = ":".join(str(p) for p in parts)
        return "recsys:" + hashlib.md5(raw.encode()).hexdigest()

    # ── Internals ───────────────────────────────────────────────

    def _raw_get(self, key: str) -> Optional[str]:
        if self._redis:
            try:
                return self._redis.get(key)
            except Exception as exc:
                logger.warning("Redis get failed", key=key, error=str(exc))
        return self._local.get(key)

    def _raw_set(self, key: str, value: str, ttl: int) -> bool:
        if self._redis:
            try:
                self._redis.setex(key, ttl, value)
                return True
            except Exception as exc:
                logger.warning("Redis set failed, falling back to local", error=str(exc))

        # Local fallback (no TTL enforcement — fine for dev)
        if len(self._local) > 1000:
            # Simple eviction: drop oldest half
            keys = list(self._local.keys())
            for k in keys[: len(keys) // 2]:
                del self._local[k]
        self._local[key] = value
        return True


# Singleton
_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
