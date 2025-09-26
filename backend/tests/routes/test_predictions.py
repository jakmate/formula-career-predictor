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
        """Test exception handling in get_predictions - covers lines 15-20."""
        with patch('app.routes.predictions.PredictionService') as mock_service_class:
            # Mock the service to raise an exception
            mock_service = Mock()
            mock_service.get_predictions = AsyncMock(side_effect=Exception("Test error"))
            mock_service_class.return_value = mock_service

            with patch('app.routes.predictions.LOGGER') as mock_logger:
                with pytest.raises(HTTPException) as exc_info:
                    # Pass both dependencies
                    await get_predictions("f1", mock_app_state, mock_data_service)

                # Verify exception details
                assert exc_info.value.status_code == 500
                assert exc_info.value.detail == "Test error"

                # Verify error was logged
                mock_logger.error.assert_called_once_with("Error in get_predictions: Test error")

                # Verify service was instantiated correctly with all 3 arguments
                mock_service_class.assert_called_once_with(mock_app_state, "f1", mock_data_service)

    @pytest.mark.asyncio
    async def test_different_exception_types(self, mock_app_state, mock_data_service):
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
