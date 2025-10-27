from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends

from app.dependencies import get_cronjob_service
from app.models.system import RefreshResponse
from app.services.cronjobs_service import CronjobService


router = APIRouter()


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_data(
    background_tasks: BackgroundTasks,
    cronjob_service: CronjobService = Depends(get_cronjob_service)
):
    """Trigger data refresh and model retraining"""
    background_tasks.add_task(cronjob_service.scrape_and_train_task)
    return RefreshResponse(
        message="Data refresh and training started in background",
        estimated_completion=datetime.now() + timedelta(minutes=2)
    )


@router.post("/refresh/predictions", response_model=RefreshResponse)
async def refresh_predictions(
    background_tasks: BackgroundTasks,
    cronjob_service: CronjobService = Depends(get_cronjob_service)
):
    """Trigger predictions refresh and model retraining"""
    background_tasks.add_task(cronjob_service.scrape_predictions)
    return RefreshResponse(
        message="Predictions refresh and training started in background",
        estimated_completion=datetime.now() + timedelta(minutes=1)
    )


@router.post("/refresh/schedule", response_model=RefreshResponse)
async def refresh_schedule(
    background_tasks: BackgroundTasks,
    cronjob_service: CronjobService = Depends(get_cronjob_service)
):
    """Trigger schedule refresh"""
    background_tasks.add_task(cronjob_service.scrape_schedule)
    return RefreshResponse(
        message="Schedule refresh started in background",
        estimated_completion=datetime.now() + timedelta(minutes=1)
    )
