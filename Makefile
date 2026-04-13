# ════════════════════════════════════════════════════════════════
#  Makefile — RecommendAI Developer Shortcuts
# ════════════════════════════════════════════════════════════════
.PHONY: help install dev-backend dev-frontend test test-cov lint \
        docker-build docker-up docker-down docker-logs clean

PYTHON = python3
PIP    = pip3

# ── Default target ────────────────────────────────────────────
help:
	@echo ""
	@echo "  RecommendAI — Available Make Targets"
	@echo "  ────────────────────────────────────"
	@echo "  make install       Install all Python dependencies"
	@echo "  make dev-backend   Run FastAPI backend (hot-reload)"
	@echo "  make dev-frontend  Run Streamlit frontend"
	@echo "  make test          Run all tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make lint          Run ruff linter"
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-up     Start all services via Docker Compose"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   Tail all service logs"
	@echo "  make clean         Remove caches and build artefacts"
	@echo ""

# ── Python setup ──────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements/all.txt
	$(PIP) install pytest pytest-cov httpx ruff

# ── Local development ─────────────────────────────────────────
dev-backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	streamlit run frontend/app.py --server.port 8501

# ── Testing ───────────────────────────────────────────────────
test:
	pytest

test-cov:
	pytest --cov=backend --cov=model --cov-report=term-missing --cov-report=html

# ── Linting ───────────────────────────────────────────────────
lint:
	ruff check backend/ model/ config/ frontend/
	ruff format --check backend/ model/ config/ frontend/

format:
	ruff format backend/ model/ config/ frontend/

# ── Docker ────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d
	@echo "Services started:"
	@echo "  Frontend  → http://localhost"
	@echo "  API Docs  → http://localhost/docs"
	@echo "  Health    → http://localhost/health"

docker-down:
	docker compose down

docker-restart:
	docker compose restart

docker-logs:
	docker compose logs -f

docker-logs-backend:
	docker compose logs -f backend

docker-logs-frontend:
	docker compose logs -f frontend

# ── Cleanup ───────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	rm -rf model/cache/*.pkl 2>/dev/null || true
	@echo "Cleaned up"
