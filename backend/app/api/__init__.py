from fastapi import APIRouter
from .routes.recommend import router as recommend_router
from .routes.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(recommend_router)
