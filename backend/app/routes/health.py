from datetime import datetime
from fastapi import APIRouter, Depends

from app.dependencies import get_app_state
from app.core.state import AppState
from app.models.system import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "name": "F3/F2 Racing Predictions API",
        "status": "running",
        "health": "/api/health"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check(app_state: AppState = Depends(get_app_state)):
    """Detailed health check with system status"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded=len(app_state.models),
        last_training=app_state.system_status.get("last_training")
    )
