# RecommendAI

> Production-grade personalized recommendation system using *Cosine Similarity* with FastAPI backend, Streamlit frontend, Redis caching, MongoDB persistence, and full Docker deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT                              │
│              Browser / API Consumer                         │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    NGINX (Port 80/443)                      │
│        /          →  Streamlit Frontend                     │
│        /api/      →  FastAPI Backend                        │
│        /docs      →  Swagger UI                             │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────────┐
│   STREAMLIT      │   │         FASTAPI BACKEND              │
│   Frontend       │   │                                      │
│   (Port 8501)    │   │  ┌─────────────────────────────┐     │
│                  │   │  │   API Routes (Controllers)  │     │
│  ┌────────────┐  │   │  │  /recommend  /search        │     │
│  │ api_client │──┼───┼─▶│  /titles     /categories   │     │
│  └────────────┘  │   │  └──────────────┬──────────────┘     │
│  ┌────────────┐  │   │                 │                    │
│  │ components │  │   │  ┌──────────────▼──────────────┐     │
│  └────────────┘  │   │  │   Recommendation Service     │    │
└──────────────────┘   │  │   (Business Logic Layer)     │    │
                       │  └──────────┬─────────┬─────────┘    │
                       │             │         │              │
                       │  ┌──────────▼──┐  ┌──▼──────────┐    │
                       │  │    Redis    │  │   MongoDB   │    │
                       │  │   Cache     │  │ Persistence │    │
                       │  └─────────────┘  └─────────────┘    │
                       │             │                        │
                       │  ┌──────────▼─────────────────────┐  │
                       │  │     ML RECOMMENDATION ENGINE    │ │
                       │  │                                 | │
                       │  │  ┌──────────────────────────┐   │ │
                       │  │  │   DataPreprocessor        │  │ │
                       │  │  │  • CSV Loading            │  │ │
                       │  │  │  • TF-IDF Vectorisation   │  │ │
                       │  │  │  • Feature Engineering    │  │ │
                       │  │  └────────────┬─────────────┘   │ │
                       │  │               │                 │ │
                       │  │  ┌────────────▼─────────────┐   │ │
                       │  │  │  ContentBasedRecommender  │  │ │
                       │  │  │  • Cosine Similarity      │  │ │
                       │  │  │  • Similarity Matrix Cache│  │ │
                       │  │  └──────────────────────────┘   │ │
                       │  │  ┌──────────────────────────┐   │ │
                       │  │  │  CollaborativeRecommender │  │ │
                       │  │  │  • User-Item Matrix       │  │ │
                       │  │  │  • User Similarity        │  │ │
                       │  │  └──────────────────────────┘   │ │
                       │  │  ┌──────────────────────────┐   │ │
                       │  │  │    HybridRecommender      │  │ │
                       │  │  │  • α×Content+(1-α)×Collab │  │ │
                       │  │  └──────────────────────────┘  │  │
                       │  └────────────────────────────────┘  │
                       └──────────────────────────────────────┘
```

---

## 📂 Project Structure

```
recsys/
│
├── frontend/                    # Streamlit UI
│   ├── app.py                   # Main Streamlit app (entry point)
│   ├── api_client.py            # HTTP client for backend API
│   └── components.py            # Reusable UI components
│
├── backend/                     # FastAPI service
│   ├── main.py                  # ASGI entry point (uvicorn target)
│   ├── app/
│   │   ├── api/
│   │   │   ├── __init__.py      # Router aggregator
│   │   │   └── routes/
│   │   │       ├── recommend.py # /recommend endpoints
│   │   │       └── health.py    # /health, / endpoints
│   │   ├── core/
│   │   │   └── application.py   # App factory + lifespan hooks
│   │   ├── middleware/
│   │   │   └── logging_middleware.py  # Logging + rate limiting
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   └── services/
│   │       ├── cache_service.py          # Redis + in-memory cache
│   │       └── recommendation_service.py # Business logic layer
│   └── tests/
│       └── test_api.py          # Integration tests
│
├── model/                       # ML recommendation engine
│   ├── src/
│   │   ├── preprocessor.py      # Data loading, cleaning, TF-IDF
│   │   ├── recommender.py       # Content / Collab / Hybrid recommenders
│   │   └── engine.py            # Singleton engine factory
│   ├── cache/                   # Persisted similarity matrix (.pkl)
│   └── tests/
│       └── test_recommender.py  # ML unit tests
│
├── data/
│   └── raw/
│       └── movies.csv           # Source dataset (replace with yours)
│
├── config/
│   ├── settings.py              # Pydantic Settings (env-driven config)
│   ├── logger.py                # Structured logging (structlog)
│   └── __init__.py
│
├── docker/
│   ├── Dockerfile.backend       # Backend image (multi-stage)
│   ├── Dockerfile.frontend      # Frontend image
│   └── streamlit_config.toml    # Streamlit server config
│
├── nginx/
│   ├── nginx.conf               # Main nginx config
│   └── conf.d/
│       └── locations.conf       # Proxy routing rules
│
├── requirements/
│   ├── backend.txt              # Backend Python deps
│   ├── frontend.txt             # Frontend Python deps
│   └── all.txt                  # Combined (for local dev)
│
├── docker-compose.yml           # Full stack orchestration
├── .env.example                 # Environment template
├── .gitignore
├── Makefile                     # Developer shortcuts
├── pyproject.toml               # Pytest + coverage config
├── deploy.sh                    # Automated VPS deployment
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerised setup)

### Option A — Local Development (No Docker)

```bash
# 1. Clone & enter the repo
git clone https://github.com/yourname/recommendai.git
cd recommendai

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
make install
# or manually:
pip install -r requirements/all.txt

# 4. Configure environment
cp .env.example .env
# Edit .env if needed (defaults work for local dev)

# 5. Start backend (in one terminal)
make dev-backend
# → API running at http://localhost:8000
# → Swagger UI at http://localhost:8000/docs

# 6. Start frontend (in another terminal)
make dev-frontend
# → Streamlit at http://localhost:8501
```

### Option B — Docker Compose (Recommended)

```bash
# 1. Copy and configure .env
cp .env.example .env

# 2. Build and start all services
make docker-up
# or:
docker compose up -d --build

# Services will be available at:
#   Frontend   → http://localhost
#   API Docs   → http://localhost/docs
#   Health     → http://localhost/health
```

### Option C — Quick one-liner (dev mode)

```bash
cp .env.example .env && docker compose up --build
```

---

## ⚙️ Configuration Reference

All settings are controlled via environment variables (`.env` file).

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` or `production` |
| `BACKEND_PORT` | `8000` | FastAPI port |
| `FRONTEND_PORT` | `8501` | Streamlit port |
| `BACKEND_URL` | `http://backend:8000` | URL frontend uses to reach backend |
| `MONGO_URI` | `mongodb://mongo:27017` | MongoDB connection string |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_TTL` | `3600` | Cache TTL in seconds |
| `TOP_N_DEFAULT` | `10` | Default recommendation count |
| `SIMILARITY_THRESHOLD` | `0.1` | Minimum cosine similarity to include |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

---

## 🌐 API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Endpoints

#### `POST /recommend/`
Get top-N recommendations for an item.

**Request:**
```json
{
  "item_title": "Inception",
  "top_n": 5,
  "category_filter": "Sci-Fi",
  "user_id": "user_42",
  "mode": "hybrid"
}
```

**Response:**
```json
{
  "query_item": "Inception",
  "mode": "hybrid",
  "total": 5,
  "cached": false,
  "recommendations": [
    {
      "title": "Interstellar",
      "similarity_score": 0.8741,
      "rank": 1,
      "genres": "Adventure|Drama|Sci-Fi",
      "director": "Christopher Nolan",
      "cast": "Matthew McConaughey|Anne Hathaway",
      "rating": 8.6,
      "votes": 1900000,
      "year": 2014,
      "poster_url": "https://..."
    }
  ]
}
```

#### `GET /recommend/titles`
Returns all item titles (for dropdown population).

#### `GET /recommend/search?q=nolan&limit=10`
Search item titles by substring.

#### `GET /recommend/categories`
Returns all available genre/category values.

#### `GET /recommend/item/{title}`
Get metadata for a specific item.

#### `POST /recommend/similarity`
Compute cosine similarity between two items.

**Request:**
```json
{ "title_a": "Inception", "title_b": "Interstellar" }
```

**Response:**
```json
{
  "title_a": "Inception",
  "title_b": "Interstellar",
  "similarity_score": 0.8741
}
```

#### `GET /health`
Service health check.

**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "engine_ready": true,
  "items_loaded": 30
}
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest model/tests/test_recommender.py -v

# Run specific test class
pytest backend/tests/test_api.py::TestRecommendations -v
```

---

## 🔧 Using Your Own Dataset

The system works with any CSV dataset. Replace `data/raw/movies.csv` with your own file.

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `title` | str | Item name (must be unique) |
| `genres` | str | Pipe-separated: `Action\|Drama` |
| `director` | str | Director name |
| `cast` | str | Pipe-separated: `Actor A\|Actor B` |
| `rating` | float | Numeric score (0–10) |
| `votes` | int | Popularity metric |
| `year` | int | Release year |
| `poster_url` | str | Optional image URL |

**Adapting for non-movie data (e.g., e-commerce products):**

1. Update column names in `model/src/preprocessor.py` → `TEXT_COLS`, `NUMERIC_COLS`
2. Update `_extract_metadata()` to return your fields
3. Update Pydantic schemas in `backend/app/models/schemas.py`
4. Update the Streamlit UI labels in `frontend/app.py`

---

## 🐳 Deployment on VPS

### Automated (recommended)

```bash
# Basic deployment (HTTP only)
chmod +x deploy.sh
./deploy.sh

# With custom domain + SSL (Let's Encrypt)
./deploy.sh --domain your-domain.com --ssl
```

The script will:
1. Install Docker, Docker Compose, UFW, fail2ban
2. Configure firewall (ports 22, 80, 443)
3. Generate a secure `SECRET_KEY`
4. Obtain Let's Encrypt certificate (if `--ssl`)
5. Build and start all services
6. Register a systemd service for auto-restart on reboot

### Manual Steps

```bash
# 1. SSH into your VPS
ssh user@your-server-ip

# 2. Clone the repo
git clone https://github.com/yourname/recommendai.git
cd recommendai

# 3. Configure environment
cp .env.example .env
nano .env   # Set APP_ENV=production and update secrets

# 4. Build and start
docker compose up -d --build

# 5. Check logs
docker compose logs -f backend
```

### Useful management commands

```bash
# View running services
docker compose ps

# Restart a service
docker compose restart backend

# View logs
docker compose logs -f

# Stop everything
docker compose down

# Update (pull new code + rebuild)
git pull && docker compose up -d --build
```

---

## 🏗️ How the ML Engine Works

### 1. Feature Engineering

For each item, we build a "feature soup" combining:
- **Text features**: genres, director, cast → TF-IDF vectors (ngram_range=(1,2))
- **Numeric features**: rating, votes, year → MinMax scaled

These are concatenated into a single feature matrix:
```
feature_matrix = hstack([tfidf_matrix × 0.7, numeric_matrix × 0.3])
```

### 2. Cosine Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(feature_matrix, feature_matrix)
# Shape: (n_items × n_items)
# sim_matrix[i][j] = similarity between item i and item j
```

Cosine similarity ranges from `0.0` (orthogonal/unrelated) to `1.0` (identical feature vectors).

### 3. Recommendation

For a query item at index `i`:
```python
scores = sim_matrix[i]           # All similarity scores for item i
scores[i] = 0                    # Exclude self
top_indices = argsort(scores)[::-1][:top_n]
```

### 4. Hybrid Mode

```
score_hybrid = α × score_content + (1-α) × score_collab
```
where `α = 0.7` by default (70% content, 30% collaborative).

### 5. Caching

The `N×N` similarity matrix is:
- Computed once and saved to `model/cache/similarity_matrix.pkl`
- Loaded from disk on subsequent startups (near-instant)
- Recommendation results cached in Redis with configurable TTL

---

## 📊 Performance Notes

| Dataset Size | Matrix Computation | Matrix Size (RAM) | Cold Start |
|---|---|---|---|
| 1,000 items | ~2 sec | ~8 MB | ~5 sec |
| 10,000 items | ~30 sec | ~800 MB | ~40 sec |
| 50,000 items | ~10 min | ~20 GB | ~15 min |

**For large datasets (>10k items):**
- Use Approximate Nearest Neighbours (FAISS, Annoy) instead of full matrix
- Use sparse cosine similarity from scipy
- Consider ALS/SVD via the `implicit` or `Surprise` library

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run tests: `make test`
4. Run linter: `make lint`
5. Open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.
