import json
import os
from datetime import datetime
from fastapi import HTTPException
import pytz

from app.config import SCHEDULE_DIR
from app.models.schedule import ScheduleRequest


class ScheduleService:

    async def get_series_schedule(self, request: ScheduleRequest):
        """Get schedule for a specific racing series with timezone conversion"""
        file_path = os.path.join(SCHEDULE_DIR, f"{request.series}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Schedule data not found")

        with open(file_path, 'r') as f:
            schedule = json.load(f)

        user_timezone = request.get_timezone()
        if user_timezone != 'UTC':
            schedule = self._convert_schedule_timezone(schedule, user_timezone)

        return schedule

    async def get_next_race(self, request: ScheduleRequest):
        """Get the next upcoming race for a series with timezone conversion.
        If no upcoming races, return the last race of the season."""
        file_path = os.path.join(SCHEDULE_DIR, f"{request.series}.json")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Schedule data not found")

        with open(file_path, 'r') as f:
            schedule = json.load(f)

        total_rounds = len(schedule)
        now = datetime.now(pytz.UTC)
        next_race = None
        next_session = None
        season_completed = True

        for race in schedule:
            # Check if this race has any future sessions
            has_future_session = False
            race_earliest_session = None

            for session_name, session_info in race['sessions'].items():
                start_str = session_info.get('start')
                if not start_str:
                    continue

                try:
                    # Handle both date-only strings (for TBC) and full datetime strings
                    if len(start_str) == 10:  # YYYY-MM-DD format (TBC sessions)
                        # Create datetime at start of day for comparison
                        session_dt = datetime.strptime(start_str, '%Y-%m-%d').replace(
                            tzinfo=pytz.UTC, hour=0, minute=0, second=0, microsecond=0
                        )
                    else:
                        # Parse full datetime string
                        session_dt = datetime.fromisoformat(start_str)
                        if session_dt.tzinfo is None:
                            session_dt = pytz.UTC.localize(session_dt)

                    if session_dt > now:
                        has_future_session = True
                        season_completed = False  # Found at least one future session
                        candidate_session = {
                            'name': session_name,
                            'date': start_str,
                            'isTBC': session_info.get('time') == 'TBC'
                        }

                        # Parse next session date safely
                        if next_session:
                            next_session_date_str = next_session['date']

                            if len(next_session_date_str) > 10:
                                next_session_dt = datetime.fromisoformat(next_session_date_str)
                            else:  # Date only
                                next_session_dt = datetime.strptime(
                                    next_session_date_str,
                                    '%Y-%m-%d'
                                ).replace(tzinfo=pytz.UTC)

                            # Ensure timezone awareness
                            next_session_dt = next_session_dt.replace(tzinfo=pytz.UTC)
                        else:
                            next_session_dt = None

                        if not next_session_dt or session_dt < next_session_dt:
                            next_session = candidate_session

                        # Track earliest session in this race for fallback
                        if not race_earliest_session or session_dt < race_earliest_session:
                            race_earliest_session = session_dt

                except (ValueError, TypeError):
                    # Skip invalid datetime strings
                    continue

            # If we found future sessions and this is our first next_race, or this race is sooner
            sessions = next_race.get('sessions', {}) if next_race else {}
            first_session_start = (
                self._parse_datetime(list(sessions.values())[0]['start']).replace(tzinfo=pytz.UTC)
                if sessions else None
            )

            if has_future_session and (
                not next_race
                or not sessions
                or (race_earliest_session and race_earliest_session < first_session_start)
            ):
                next_race = race.copy()
                next_race['totalRounds'] = total_rounds

        # If no race with future sessions found, find the next race by date only
        if not next_race:
            race_start_dates = []
            for race in schedule:
                # Get the earliest date from any session in this race
                earliest_date = None
                for session_name, session_info in race['sessions'].items():
                    start_str = session_info.get('start')
                    if start_str:
                        try:
                            # Handle date-only strings (TBC sessions)
                            session_date = self._parse_datetime(start_str)
                            if session_date > now:
                                season_completed = False  # Found at least one future session
                                if not earliest_date or session_date < earliest_date:
                                    earliest_date = session_date
                        except (ValueError, TypeError):
                            continue

                if earliest_date:
                    race_start_dates.append((race, earliest_date))

            # Sort by date and pick the earliest
            if race_start_dates:
                race_start_dates.sort(key=lambda x: x[1])
                next_race = race_start_dates[0][0].copy()
                next_race['totalRounds'] = total_rounds

        # If no future races found, return the last race of the season
        if not next_race and schedule:
            next_race = schedule[-1].copy()  # Last race in the schedule
            next_race['totalRounds'] = total_rounds
            next_race['seasonCompleted'] = True
        elif next_race:
            next_race['seasonCompleted'] = season_completed

        # Apply timezone conversion and set next session
        if next_race:
            if not next_race.get('seasonCompleted') and next_session:
                next_race['nextSession'] = next_session

            user_timezone = request.get_timezone()
            if user_timezone != 'UTC':
                next_race = self._convert_race_timezone(next_race, user_timezone)

        return next_race

    def _parse_datetime(self, date_string: str) -> datetime:
        """Parse a date string that could be either YYYY-MM-DD or full ISO format"""
        if len(date_string) == 10:
            return datetime.strptime(date_string, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        else:
            dt = datetime.fromisoformat(date_string)
            if dt.tzinfo is None:
                dt = pytz.UTC.localize(dt)
            return dt

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
                if time_str and len(time_str) > 10:
                    utc_dt = datetime.fromisoformat(time_str)
                    if utc_dt.tzinfo is None:
                        utc_dt = utc_tz.localize(utc_dt)
                    local_dt = utc_dt.astimezone(target_tz)
                    session_info[time_field] = local_dt.isoformat()

        if 'nextSession' in race:
            start_str = race['nextSession'].get('date')
            if start_str and len(start_str) > 10:
                utc_dt = datetime.fromisoformat(start_str)
                if utc_dt.tzinfo is None:
                    utc_dt = utc_tz.localize(utc_dt)
                local_dt = utc_dt.astimezone(target_tz)
                race['nextSession']['date'] = local_dt.isoformat()

        return race
