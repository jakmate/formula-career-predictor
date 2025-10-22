from fastapi import APIRouter, HTTPException, Header, Query
from typing import Optional

from app.services.schedule_service import ScheduleService, ScheduleRequest
from app.config import LOGGER

router = APIRouter()


@router.get("/{series}")
async def get_series_schedule(
    series: str,
    timezone: Optional[str] = Query(None),
    x_timezone: Optional[str] = Header(None)
):
    """Get schedule for a specific racing series with timezone conversion"""
    try:
        request = ScheduleRequest(
            series=series,
            timezone=timezone,
            x_timezone=x_timezone
        )
        schedule_service = ScheduleService()
        return await schedule_service.get_series_schedule(request)
    except Exception as e:
        LOGGER.error(f"Error getting schedule for {series}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{series}/next")
async def get_next_race(
    series: str,
    timezone: Optional[str] = Query(None),
    x_timezone: Optional[str] = Header(None)
):
    """Get the next upcoming race for a series with timezone conversion"""
    try:
        request = ScheduleRequest(
            series=series,
            timezone=timezone,
            x_timezone=x_timezone
        )
        schedule_service = ScheduleService()
        return await schedule_service.get_next_race(request)
    except Exception as e:
        LOGGER.error(f"Error getting next race for {series}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
