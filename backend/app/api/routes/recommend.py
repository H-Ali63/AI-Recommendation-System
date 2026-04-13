"""
backend/app/api/routes/recommend.py
────────────────────────────────────
All /recommend endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from backend.app.models.schemas import (
    CategoriesResponse,
    ItemMetadataResponse,
    RecommendRequest,
    RecommendResponse,
    SearchRequest,
    SimilarityRequest,
    SimilarityResponse,
)
from backend.app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)

router = APIRouter(prefix="/recommend", tags=["Recommendations"])


@router.post(
    "/",
    response_model=RecommendResponse,
    summary="Get item recommendations",
    description="Returns top-N recommendations for the given item using cosine similarity.",
)
async def get_recommendations(
    body: RecommendRequest,
    svc: RecommendationService = Depends(get_recommendation_service),
) -> RecommendResponse:
    try:
        return svc.get_recommendations(
            item_title=body.item_title,
            top_n=body.top_n,
            category_filter=body.category_filter,
            user_id=body.user_id,
            mode=body.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation engine error: {exc}",
        )


@router.get(
    "/search",
    response_model=list[str],
    summary="Search item titles",
)
async def search_items(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50),
    svc: RecommendationService = Depends(get_recommendation_service),
) -> list[str]:
    return svc.search_items(q, limit)


@router.get(
    "/titles",
    response_model=list[str],
    summary="Get all item titles",
)
async def get_all_titles(
    svc: RecommendationService = Depends(get_recommendation_service),
) -> list[str]:
    return svc.get_all_titles()


@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="Get all categories / genres",
)
async def get_categories(
    svc: RecommendationService = Depends(get_recommendation_service),
) -> CategoriesResponse:
    cats = svc.get_categories()
    return CategoriesResponse(categories=cats, total=len(cats))


@router.get(
    "/item/{title}",
    response_model=ItemMetadataResponse,
    summary="Get metadata for a specific item",
)
async def get_item_metadata(
    title: str,
    svc: RecommendationService = Depends(get_recommendation_service),
) -> ItemMetadataResponse:
    try:
        meta = svc.get_item_metadata(title)
        return ItemMetadataResponse(**meta)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Get cosine similarity between two items",
)
async def get_similarity(
    body: SimilarityRequest,
    svc: RecommendationService = Depends(get_recommendation_service),
) -> SimilarityResponse:
    try:
        score = svc.get_similarity(body.title_a, body.title_b)
        return SimilarityResponse(
            title_a=body.title_a,
            title_b=body.title_b,
            similarity_score=score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
