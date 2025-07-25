import pytest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routes.schedule import router, get_series_schedule, get_next_race


@pytest.fixture
def mock_schedule_service():
    mock_service = AsyncMock()
    mock_service.get_series_schedule = AsyncMock()
    mock_service.get_next_race = AsyncMock()
    return mock_service


@pytest.fixture
def test_app():
    app = FastAPI()
    app.include_router(router)
    return app


class TestGetSeriesSchedule:
    @pytest.mark.asyncio
    async def test_get_series_schedule_success(self, mock_schedule_service):
        """Test successful schedule retrieval"""
        expected_data = {"schedule": "test_data"}
        mock_schedule_service.get_series_schedule.return_value = expected_data

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            result = await get_series_schedule("f1", "UTC", "America/New_York")

            assert result == expected_data
            mock_schedule_service.get_series_schedule.assert_called_once_with(
                "f1", "UTC", "America/New_York"
            )

    @pytest.mark.asyncio
    async def test_get_series_schedule_with_none_params(self, mock_schedule_service):
        expected_data = {"schedule": "test_data"}
        mock_schedule_service.get_series_schedule.return_value = expected_data

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            result = await get_series_schedule("f2", None, None)

            assert result == expected_data
            mock_schedule_service.get_series_schedule.assert_called_once_with(
                "f2", None, None
            )

    @pytest.mark.asyncio
    async def test_get_series_schedule_exception(self, mock_schedule_service):
        mock_schedule_service.get_series_schedule.side_effect = Exception("Service error")

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service), \
             patch('app.routes.schedule.LOGGER') as mock_logger:

            with pytest.raises(HTTPException) as exc_info:
                await get_series_schedule("f1", None, None)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Service error"
            mock_logger.error.assert_called_once()


class TestGetNextRace:
    @pytest.mark.asyncio
    async def test_get_next_race_success(self, mock_schedule_service):
        expected_data = {"next_race": "test_data"}
        mock_schedule_service.get_next_race.return_value = expected_data

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            result = await get_next_race("f1", "UTC", "America/New_York")

            assert result == expected_data
            mock_schedule_service.get_next_race.assert_called_once_with(
                "f1", "UTC", "America/New_York"
            )

    @pytest.mark.asyncio
    async def test_get_next_race_with_none_params(self, mock_schedule_service):
        expected_data = {"next_race": "test_data"}
        mock_schedule_service.get_next_race.return_value = expected_data

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            result = await get_next_race("f3", None, None)

            assert result == expected_data
            mock_schedule_service.get_next_race.assert_called_once_with(
                "f3", None, None
            )

    @pytest.mark.asyncio
    async def test_get_next_race_exception(self, mock_schedule_service):
        mock_schedule_service.get_next_race.side_effect = Exception("Service error")

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service), \
             patch('app.routes.schedule.LOGGER') as mock_logger:

            with pytest.raises(HTTPException) as exc_info:
                await get_next_race("f1", None, None)

            assert exc_info.value.status_code == 500
            assert exc_info.value.detail == "Service error"
            mock_logger.error.assert_called_once()


class TestRouterIntegration:
    def test_get_series_schedule_endpoint(self, test_app, mock_schedule_service):
        mock_schedule_service.get_series_schedule.return_value = {"schedule": "data"}

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            client = TestClient(test_app)
            response = client.get("/f1?timezone=UTC")

            assert response.status_code == 200
            assert response.json() == {"schedule": "data"}

    def test_get_next_race_endpoint(self, test_app, mock_schedule_service):
        mock_schedule_service.get_next_race.return_value = {"next_race": "data"}

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            client = TestClient(test_app)
            response = client.get("/f1/next?timezone=UTC")

            assert response.status_code == 200
            assert response.json() == {"next_race": "data"}

    def test_header_timezone_handling(self, test_app, mock_schedule_service):
        mock_schedule_service.get_series_schedule.return_value = {"schedule": "data"}

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            client = TestClient(test_app)
            response = client.get("/f1", headers={"X-Timezone": "Europe/London"})

            assert response.status_code == 200
            mock_schedule_service.get_series_schedule.assert_called_once_with(
                "f1", None, "Europe/London"
            )

    def test_both_timezone_params(self, test_app, mock_schedule_service):
        mock_schedule_service.get_series_schedule.return_value = {"schedule": "data"}

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            client = TestClient(test_app)
            response = client.get(
                "/f1?timezone=UTC",
                headers={"X-Timezone": "Europe/London"}
            )

            assert response.status_code == 200
            mock_schedule_service.get_series_schedule.assert_called_once_with(
                "f1", "UTC", "Europe/London"
            )

    def test_error_response_format(self, test_app, mock_schedule_service):
        mock_schedule_service.get_series_schedule.side_effect = Exception("Test error")

        with patch('app.routes.schedule.ScheduleService', return_value=mock_schedule_service):
            client = TestClient(test_app)
            response = client.get("/f1")

            assert response.status_code == 500
            assert response.json() == {"detail": "Test error"}
