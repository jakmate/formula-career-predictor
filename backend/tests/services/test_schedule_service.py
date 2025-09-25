import pytest
import json
from datetime import datetime
from unittest.mock import patch, mock_open
from fastapi import HTTPException
import pytz

from app.services.schedule_service import ScheduleService


class TestScheduleService:

    @pytest.fixture
    def schedule_service(self):
        return ScheduleService()

    @pytest.fixture
    def sample_schedule_data(self):
        return [
            {
                "round": 1,
                "name": "Melbourne",
                "location": "Australia",
                "sessions": {
                    "practice": {
                        "start": "2025-03-14T01:00:00",
                        "end": "2025-03-14T01:45:00"
                    },
                    "qualifying": {
                        "start": "2025-03-14T04:50:00",
                        "end": "2025-03-14T05:20:00"
                    },
                    "sprint": {
                        "start": "2025-03-15T01:25:00",
                        "end": "2025-03-15T02:05:00"
                    },
                    "race": {
                        "start": "2025-03-15T22:55:00",
                        "end": "2025-03-15T23:40:00"
                    }
                }
            },
            {
                "round": 2,
                "name": "Sakhir",
                "location": "Bahrain",
                "sessions": {
                    "practice": {
                        "start": "2025-04-11T11:00:00",
                        "end": "2025-04-11T11:45:00"
                    },
                    "qualifying": {
                        "start": "2025-04-11T14:55:00",
                        "end": "2025-04-11T15:25:00"
                    },
                    "sprint": {
                        "start": "2025-04-12T11:15:00",
                        "end": "2025-04-12T11:55:00"
                    },
                    "race": {
                        "start": "2025-04-13T07:40:00",
                        "end": "2025-04-13T08:25:00"
                    }
                }
            }
        ]

    @pytest.fixture
    def sample_schedule_with_tbc(self):
        return [
            {
                "round": 9,
                "name": "Budapest",
                "location": "Hungary",
                "sessions": {
                    "practice": {
                        "start": "2025-08-01",
                        "time": "TBC"
                    },
                    "qualifying": {
                        "start": "2025-08-01",
                        "time": "TBC"
                    },
                    "sprint": {
                        "start": "2025-08-02",
                        "time": "TBC"
                    },
                    "race": {
                        "start": "2025-08-03T15:00:00",
                        "end": "2025-08-03T16:00:00"
                    }
                }
            }
        ]

    # Tests for get_series_schedule
    @pytest.mark.asyncio
    async def test_get_series_schedule_invalid_series(self, schedule_service):
        with pytest.raises(HTTPException) as exc_info:
            await schedule_service.get_series_schedule("invalid_series")

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Invalid series specified"

    @pytest.mark.asyncio
    async def test_get_series_schedule_file_not_found(self, schedule_service):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await schedule_service.get_series_schedule("f1")

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Schedule data not found"

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_series_schedule_success_utc(self, schedule_service, sample_schedule_data):
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                result = await schedule_service.get_series_schedule("f1")

                assert result == sample_schedule_data

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_series_schedule_with_timezone_param(self, schedule_service, sample_schedule_data): # noqa: 501
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                with patch.object(schedule_service, '_convert_schedule_timezone') as mock_convert:
                    mock_convert.return_value = sample_schedule_data

                    await schedule_service.get_series_schedule("f1", timezone="America/New_York")

                    mock_convert.assert_called_once_with(sample_schedule_data, "America/New_York")

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_series_schedule_with_x_timezone_param(self, schedule_service, sample_schedule_data): # noqa: 501
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                with patch.object(schedule_service, '_convert_schedule_timezone') as mock_convert:
                    mock_convert.return_value = sample_schedule_data

                    await schedule_service.get_series_schedule("f1", x_timezone="Europe/London")

                    mock_convert.assert_called_once_with(sample_schedule_data, "Europe/London")

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_series_schedule_timezone_precedence(self, schedule_service, sample_schedule_data): # noqa: 501
        """Test that timezone parameter takes precedence over x_timezone"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                with patch.object(schedule_service, '_convert_schedule_timezone') as mock_convert:
                    mock_convert.return_value = sample_schedule_data

                    await schedule_service.get_series_schedule(
                        "f1",
                        timezone="America/New_York",
                        x_timezone="Europe/London"
                    )

                    mock_convert.assert_called_once_with(sample_schedule_data, "America/New_York")

    # Tests for get_next_race
    @pytest.mark.asyncio
    async def test_get_next_race_invalid_series(self, schedule_service):
        with pytest.raises(HTTPException) as exc_info:
            await schedule_service.get_next_race("invalid_series")

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Invalid series specified"

    @pytest.mark.asyncio
    async def test_get_next_race_file_not_found(self, schedule_service):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(HTTPException) as exc_info:
                await schedule_service.get_next_race("f1")

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Schedule data not found"

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_success(self, schedule_service, sample_schedule_data):
        # Set all sessions in first race to past, only first session of second race to future
        future_time = "2030-04-11T11:00:00"

        # All sessions in first race to past
        sample_schedule_data[0]["sessions"]["practice"]["start"] = "2020-01-01T11:30:00"
        sample_schedule_data[0]["sessions"]["qualifying"]["start"] = "2020-01-01T11:30:00"
        sample_schedule_data[0]["sessions"]["sprint"]["start"] = "2020-01-01T11:30:00"
        sample_schedule_data[0]["sessions"]["race"]["start"] = "2020-01-01T11:30:00"

        # Only practice in second race to future
        sample_schedule_data[1]["sessions"]["practice"]["start"] = future_time
        sample_schedule_data[1]["sessions"]["qualifying"]["start"] = "2020-01-01T11:30:00"
        sample_schedule_data[1]["sessions"]["sprint"]["start"] = "2020-01-01T11:30:00"
        sample_schedule_data[1]["sessions"]["race"]["start"] = "2020-01-01T11:30:00"

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                result = await schedule_service.get_next_race("f1")

                assert result is not None
                assert result["round"] == 2
                assert result["totalRounds"] == 2
                assert "nextSession" in result
                assert result["nextSession"]["name"] == "practice"
                assert result["nextSession"]["date"] == future_time
                assert result.get("seasonCompleted") is False

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_returns_last_race(self, schedule_service, sample_schedule_data):
        # All sessions in the past
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                result = await schedule_service.get_next_race("f1")

                # Should return last race of the season
                assert result is not None
                assert result["round"] == 2  # Last race in the schedule
                assert result["totalRounds"] == 2
                assert result.get("seasonCompleted") is True
                assert "nextSession" not in result

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_empty_schedule(self, schedule_service):
        empty_schedule = []

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(empty_schedule))):
                result = await schedule_service.get_next_race("f1")

                assert result is None

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_with_tbc_sessions(self, schedule_service, sample_schedule_with_tbc): # noqa: 501
        future_time = "2030-08-03T15:00:00"
        sample_schedule_with_tbc[0]["sessions"]["race"]["start"] = future_time

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_with_tbc))):
                result = await schedule_service.get_next_race("f1")

                assert result is not None
                assert result["nextSession"]["name"] == "race"
                assert result["nextSession"]["date"] == future_time
                assert result.get("seasonCompleted") is False

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_with_tbc_date_only_sessions(self, schedule_service):
        # Test TBC sessions with date-only format
        tbc_schedule = [
            {
                "round": 1,
                "name": "Test GP",
                "location": "Test",
                "sessions": {
                    "practice": {
                        "start": "2030-08-01",  # Date-only format for TBC
                        "time": "TBC"
                    },
                    "race": {
                        "start": "2030-08-02T15:00:00"
                    }
                }
            }
        ]

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(tbc_schedule))):
                result = await schedule_service.get_next_race("f1")

                assert result is not None
                assert result["nextSession"]["name"] == "practice"
                assert result["nextSession"]["date"] == "2030-08-01"
                assert result["nextSession"]["isTBC"] is True

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_with_timezone(self, schedule_service, sample_schedule_data):
        future_time = "2030-03-14T01:00:00"
        sample_schedule_data[0]["sessions"]["practice"]["start"] = future_time

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_schedule_data))):
                with patch.object(schedule_service, '_convert_race_timezone') as mock_convert:
                    mock_convert.return_value = sample_schedule_data[0]

                    await schedule_service.get_next_race("f1", timezone="America/New_York")

                    mock_convert.assert_called_once()

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_sessions_without_start(self, schedule_service):
        schedule_without_start = [
            {
                "round": 1,
                "name": "Test Grand Prix",
                "location": "Test",
                "sessions": {
                    "practice": {
                        "end": "2030-03-14T02:00:00"
                    }
                }
            }
        ]

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(schedule_without_start))):
                result = await schedule_service.get_next_race("f1")

                # Should return the last race with seasonCompleted=True
                assert result is not None
                assert result["round"] == 1
                assert result.get("seasonCompleted") is True

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_earliest_session_selection(self, schedule_service):
        # Test that earliest session is selected as next session
        test_schedule = [
            {
                "round": 1,
                "name": "Test GP",
                "location": "Test",
                "sessions": {
                    "qualifying": {
                        "start": "2030-03-15T14:00:00"  # Later
                    },
                    "practice": {
                        "start": "2030-03-15T10:00:00"  # Earlier - should be selected
                    },
                    "race": {
                        "start": "2030-03-16T15:00:00"
                    }
                }
            }
        ]

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(test_schedule))):
                result = await schedule_service.get_next_race("f1")

                assert result is not None
                assert result["nextSession"]["name"] == "practice"
                assert result["nextSession"]["date"] == "2030-03-15T10:00:00"

    @pytest.mark.asyncio
    @patch('app.config.SCHEDULE_DIR', '/test/schedules')
    async def test_get_next_race_invalid_datetime_handling(self, schedule_service):
        # Test handling of invalid datetime strings
        invalid_schedule = [
            {
                "round": 1,
                "name": "Test GP",
                "location": "Test",
                "sessions": {
                    "practice": {
                        "start": "invalid-datetime"
                    },
                    "race": {
                        "start": "2030-03-16T15:00:00"
                    }
                }
            }
        ]

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(invalid_schedule))):
                result = await schedule_service.get_next_race("f1")

                assert result is not None
                assert result["nextSession"]["name"] == "race"  # Should skip invalid datetime

    # Tests for _parse_datetime
    def test_parse_datetime_date_only(self, schedule_service):
        result = schedule_service._parse_datetime("2025-08-01")
        expected = datetime(2025, 8, 1, tzinfo=pytz.UTC)
        assert result == expected

    def test_parse_datetime_full_iso(self, schedule_service):
        result = schedule_service._parse_datetime("2025-08-01T15:30:00")
        expected = datetime(2025, 8, 1, 15, 30, tzinfo=pytz.UTC)
        assert result == expected

    def test_parse_datetime_with_timezone(self, schedule_service):
        result = schedule_service._parse_datetime("2025-08-01T15:30:00+02:00")
        # Should preserve timezone info
        assert result.tzinfo is not None
        assert result.year == 2025
        assert result.month == 8
        assert result.day == 1

    # Tests for _convert_schedule_timezone
    def test_convert_schedule_timezone(self, schedule_service, sample_schedule_data):
        result = schedule_service._convert_schedule_timezone(sample_schedule_data, "America/New_York") # noqa: 501

        # Check that times were converted (specific time conversion depends on timezone)
        assert result[0]["sessions"]["practice"]["start"] != "2025-03-14T01:00:00"
        assert result[0]["sessions"]["practice"]["end"] != "2025-03-14T01:45:00"

    def test_convert_schedule_timezone_with_tbc(self, schedule_service, sample_schedule_with_tbc):
        # Should not fail with TBC sessions
        result = schedule_service._convert_schedule_timezone(sample_schedule_with_tbc, "America/New_York") # noqa: 501

        assert result[0]["sessions"]["practice"]["time"] == "TBC"

    def test_convert_schedule_timezone_missing_time_fields(self, schedule_service):
        schedule_missing_times = [
            {
                "round": 1,
                "name": "Test Grand Prix",
                "location": "Test",
                "sessions": {
                    "practice": {
                        # Missing start and end
                    }
                }
            }
        ]

        # Should not fail
        result = schedule_service._convert_schedule_timezone(schedule_missing_times, "America/New_York") # noqa: 501
        assert len(result) == 1

    # Tests for _convert_race_timezone
    def test_convert_race_timezone(self, schedule_service, sample_schedule_data):
        race = sample_schedule_data[0].copy()

        result = schedule_service._convert_race_timezone(race, "America/New_York")

        # Check that times were converted
        assert result["sessions"]["practice"]["start"] != "2025-03-14T01:00:00"
        assert result["sessions"]["practice"]["end"] != "2025-03-14T01:45:00"

    def test_convert_race_timezone_with_next_session(self, schedule_service, sample_schedule_data):
        race = sample_schedule_data[0].copy()
        race["nextSession"] = {
            "name": "practice",
            "date": "2025-03-14T01:00:00"
        }

        result = schedule_service._convert_race_timezone(race, "America/New_York")

        # Check that nextSession date was converted
        assert result["nextSession"]["date"] != "2025-03-14T01:00:00"

    def test_convert_race_timezone_with_next_session_date_only(self, schedule_service,
                                                               sample_schedule_data):
        race = sample_schedule_data[0].copy()
        race["nextSession"] = {
            "name": "practice",
            "date": "2025-03-14"  # Date-only format
        }

        result = schedule_service._convert_race_timezone(race, "America/New_York")

        # Should not convert date-only formats
        assert result["nextSession"]["date"] == "2025-03-14"

    def test_convert_race_timezone_with_tbc_sessions(self, schedule_service):
        race_with_tbc = {
            "round": 9,
            "name": "Budapest",
            "location": "Hungary",
            "sessions": {
                "practice": {
                    "start": "2025-08-01",
                    "time": "TBC"
                }
            }
        }

        # Should not fail with TBC sessions
        result = schedule_service._convert_race_timezone(race_with_tbc, "America/New_York")
        assert result["sessions"]["practice"]["time"] == "TBC"

    def test_convert_race_timezone_missing_next_session_date(self, schedule_service, sample_schedule_data): # noqa: 501
        race = sample_schedule_data[0].copy()
        race["nextSession"] = {
            "name": "practice1"
            # Missing date
        }

        # Should not fail
        result = schedule_service._convert_race_timezone(race, "America/New_York")
        assert "nextSession" in result
