import json
import os
from datetime import datetime
from typing import Optional
from fastapi import HTTPException
import pytz

from app.config import SCHEDULE_DIR


class ScheduleService:

    async def get_series_schedule(
        self,
        series: str,
        timezone: Optional[str] = None,
        x_timezone: Optional[str] = None
    ):
        """Get schedule for a specific racing series with timezone conversion"""
        if series not in ['f1', 'f2', 'f3']:
            raise HTTPException(status_code=404, detail="Invalid series specified")

        file_path = os.path.join(SCHEDULE_DIR, f"{series}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Schedule data not found")

        with open(file_path, 'r') as f:
            schedule = json.load(f)

        user_timezone = timezone or x_timezone or 'UTC'
        if user_timezone != 'UTC':
            schedule = self._convert_schedule_timezone(schedule, user_timezone)

        return schedule

    async def get_next_race(
        self,
        series: str,
        timezone: Optional[str] = None,
        x_timezone: Optional[str] = None
    ):
        """Get the next upcoming race for a series with timezone conversion"""
        if series not in ['f1', 'f2', 'f3']:
            raise HTTPException(status_code=404, detail="Invalid series specified")

        file_path = os.path.join(SCHEDULE_DIR, f"{series}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Schedule data not found")

        with open(file_path, 'r') as f:
            schedule = json.load(f)

        total_rounds = len(schedule)
        now = datetime.utcnow()
        next_race = None
        next_session = None

        for race in schedule:
            for session_name, session_info in race['sessions'].items():
                if session_info.get('time') == 'TBC':
                    continue

                start_str = session_info.get('start')
                if not start_str:
                    continue

                session_dt = datetime.fromisoformat(start_str)
                if session_dt > now:
                    candidate_session = {'name': session_name, 'date': start_str}
                    if (not next_session or session_dt <
                            datetime.fromisoformat(next_session['date'])):
                        next_session = candidate_session
                        if not next_race:
                            next_race = race
                            next_race['totalRounds'] = total_rounds

        if next_race and next_session:
            next_race['nextSession'] = next_session
            user_timezone = timezone or x_timezone or 'UTC'
            if user_timezone != 'UTC':
                next_race = self._convert_race_timezone(next_race, user_timezone)

        return next_race

    def _convert_schedule_timezone(self, schedule, target_timezone):
        """Convert all datetime strings in schedule from UTC to target timezone"""
        target_tz = pytz.timezone(target_timezone)
        utc_tz = pytz.UTC

        for race in schedule:
            for session_name, session_info in race['sessions'].items():
                if session_info.get('time') == 'TBC':
                    continue

                for time_field in ['start', 'end']:
                    time_str = session_info.get(time_field)
                    if time_str:
                        utc_dt = datetime.fromisoformat(time_str)
                        if utc_dt.tzinfo is None:
                            utc_dt = utc_tz.localize(utc_dt)
                        local_dt = utc_dt.astimezone(target_tz)
                        session_info[time_field] = local_dt.isoformat()

        return schedule

    def _convert_race_timezone(self, race, target_timezone):
        """Convert datetime strings in a single race from UTC to target timezone"""
        target_tz = pytz.timezone(target_timezone)
        utc_tz = pytz.UTC

        for session_name, session_info in race['sessions'].items():
            if session_info.get('time') == 'TBC':
                continue

            for time_field in ['start', 'end']:
                time_str = session_info.get(time_field)
                if time_str:
                    utc_dt = datetime.fromisoformat(time_str)
                    if utc_dt.tzinfo is None:
                        utc_dt = utc_tz.localize(utc_dt)
                    local_dt = utc_dt.astimezone(target_tz)
                    session_info[time_field] = local_dt.isoformat()

        if 'nextSession' in race:
            start_str = race['nextSession'].get('date')
            if start_str:
                utc_dt = datetime.fromisoformat(start_str)
                if utc_dt.tzinfo is None:
                    utc_dt = utc_tz.localize(utc_dt)
                local_dt = utc_dt.astimezone(target_tz)
                race['nextSession']['date'] = local_dt.isoformat()

        return race
