from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from app.services.schedule_service import ScheduleService
from app.config import LOGGER

router = APIRouter()


@router.get("/{series}")
async def get_series_schedule(
    series: str,
    timezone: Optional[str] = None,
    x_timezone: Optional[str] = Header(None)
):
    """Get schedule for a specific racing series with timezone conversion"""
    try:
        schedule_service = ScheduleService()
        return await schedule_service.get_series_schedule(series, timezone, x_timezone)
    except Exception as e:
        LOGGER.error(f"Error getting schedule for {series}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{series}/next")
async def get_next_race(
    series: str,
    timezone: Optional[str] = None,
    x_timezone: Optional[str] = Header(None)
):
    """Get the next upcoming race for a series with timezone conversion"""
    try:
        schedule_service = ScheduleService()
        return await schedule_service.get_next_race(series, timezone, x_timezone)
    except Exception as e:
        LOGGER.error(f"Error getting next race for {series}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
