from fastapi import APIRouter, Depends, HTTPException

from app.services.prediction_service import PredictionService
from app.models.predictions import AllPredictionsResponse
from app.dependencies import get_app_state
from app.core.state import AppState
from app.config import LOGGER

router = APIRouter()


@router.get("", response_model=AllPredictionsResponse)
async def get_all_predictions(app_state: AppState = Depends(get_app_state)):
    """Get predictions from all models"""
    try:
        prediction_service = PredictionService(app_state)
        return await prediction_service.get_all_predictions()
    except Exception as e:
        LOGGER.error(f"Error in get_all_predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
