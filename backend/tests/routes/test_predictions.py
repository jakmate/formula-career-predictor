import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import HTTPException

from app.routes.predictions import get_predictions


class TestGetPredictions:
    @pytest.fixture
    def mock_app_state(self):
        """Mock app state."""
        return Mock()

    @pytest.fixture
    def mock_data_service(self):
        """Mock data service."""
        return Mock()

    @pytest.mark.asyncio
    async def test_prediction_service_exception_handling(self, mock_app_state, mock_data_service):
        """Test different exception types are handled properly."""
        exceptions_to_test = [
            ValueError("Invalid value"),
            KeyError("Missing key"),
            RuntimeError("Runtime issue")
        ]

        for exception in exceptions_to_test:
            with patch('app.routes.predictions.PredictionService') as mock_service_class:
                mock_service = Mock()
                mock_service.get_predictions = AsyncMock(side_effect=exception)
                mock_service_class.return_value = mock_service

                with patch('app.routes.predictions.LOGGER'):
                    with pytest.raises(HTTPException) as exc_info:
                        await get_predictions("f2_to_f1", mock_app_state, mock_data_service)

                    assert exc_info.value.status_code == 500
                    assert exc_info.value.detail == str(exception)
