from datetime import datetime, timedelta
from fastapi import APIRouter, BackgroundTasks, Depends

from app.dependencies import get_scheduler_service
from app.models.system import RefreshResponse
from app.services.cronjobs_service import CronjobService


router = APIRouter()


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_data(
    background_tasks: BackgroundTasks,
    scheduler_service: CronjobService = Depends(get_scheduler_service)
):
    """Trigger data refresh and model retraining"""
    background_tasks.add_task(scheduler_service.scrape_and_train_task)
    return RefreshResponse(
        message="Data refresh and training started in background",
        estimated_completion=datetime.now() + timedelta(minutes=2)
    )
