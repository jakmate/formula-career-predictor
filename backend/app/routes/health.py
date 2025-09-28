from datetime import datetime
from fastapi import APIRouter, Depends, Response, status, Request

from app.dependencies import get_app_state
from app.core.state import AppState
from app.models.system import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", tags=["Health"])
async def root(request: Request):
    """Health check endpoint"""
    return {
        "name": "Formula Predictions API",
        "status": "running",
        "health": str(request.app.url_path_for("health_check"))
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(app_state: AppState = Depends(get_app_state)):
    """Health check with system status"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=len(app_state.models),
        last_training=app_state.system_status.get("last_training")
    )


@router.head("/health")
async def health_head():
    return Response(status_code=status.HTTP_200_OK)
