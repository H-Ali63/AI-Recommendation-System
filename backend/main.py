"""
backend/main.py
───────────────
FastAPI entry point.

Run locally:
    uvicorn backend.main:app --reload --port 8000

Production (via Docker):
    gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.application import create_app
from config.settings import settings

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=not settings.is_production,
        log_level=settings.LOG_LEVEL.lower(),
    )
