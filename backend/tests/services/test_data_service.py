import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from fastapi import HTTPException

from app.services.data_service import DataService
from app.core.state import AppState
from app.config import CURRENT_YEAR

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app_state():
    return Mock(spec=AppState)


@pytest.fixture
def data_service(mock_app_state):
    return DataService(mock_app_state)


class TestDataService:

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.engineer_features')
    async def test_load_current_data_empty_feeder_df(self, mock_engineer, mock_target,
                                                     mock_quali_features, mock_load_quali,
                                                     mock_load_standings, mock_load_data,
                                                     data_service):
        """Test HTTPException when feeder_df is empty"""
        mock_load_data.return_value = pd.DataFrame()  # Empty dataframe

        with pytest.raises(HTTPException) as exc_info:
            await data_service.load_current_data('f2_to_f1')

        assert exc_info.value.status_code == 404
        assert "No F2 data available" in str(exc_info.value.detail)

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.engineer_features')
    async def test_load_current_data_fallback_to_max_year(self, mock_engineer, mock_target,
                                                          mock_quali_features, mock_load_quali,
                                                          mock_load_standings, mock_load_data,
                                                          data_service):
        """Test fallback when current year data is empty"""
        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_standings.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_quali.return_value = pd.DataFrame({'driver': ['A']})
        mock_quali_features.return_value = pd.DataFrame({'driver': ['A']})
        mock_target.return_value = pd.DataFrame({'driver': ['A'], 'promoted': [1]})

        # Features dataframe with no current year data but has historical data
        features_df = pd.DataFrame({
            'driver': ['A', 'B'],
            'year': [2022, 2023],  # No current year data
            'feature1': [1, 2]
        })
        mock_engineer.return_value = features_df

        result = await data_service.load_current_data('f2_to_f1')

        # Should return data from max year (2023)
        assert len(result) == 1
        assert result['year'].iloc[0] == 2023

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.engineer_features')
    async def test_load_current_data_no_drivers_available(self, mock_engineer, mock_target,
                                                          mock_quali_features, mock_load_quali,
                                                          mock_load_standings, mock_load_data,
                                                          data_service):
        """Test HTTPException when no drivers data available"""
        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_standings.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_quali.return_value = pd.DataFrame({'driver': ['A']})
        mock_quali_features.return_value = pd.DataFrame({'driver': ['A']})
        mock_target.return_value = pd.DataFrame({'driver': ['A'], 'promoted': [1]})

        # Features dataframe with no current year data and empty after max year fallback
        features_df = pd.DataFrame(columns=['year', 'feature1'])  # Empty with columns
        mock_engineer.return_value = features_df

        with pytest.raises(HTTPException) as exc_info:
            await data_service.load_current_data('f2_to_f1')

        assert exc_info.value.status_code == 404
        assert "No drivers data available" in str(exc_info.value.detail)

    def test_parse_series_unknown_series(self, data_service):
        """Test ValueError for unknown series"""
        with pytest.raises(ValueError) as exc_info:
            data_service._parse_series('unknown_series')

        assert "Unknown series: unknown_series" in str(exc_info.value)

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.engineer_features')
    @patch('app.services.model_service.ModelService')
    @patch('app.services.prediction_service.PredictionService')
    @patch('app.services.data_service.LOGGER')
    async def test_initialize_system_full_flow(self, mock_logger, mock_pred_service_class,
                                               mock_model_service_class, mock_engineer,
                                               mock_target, mock_quali_features, mock_load_quali,
                                               mock_load_standings, mock_load_data, data_service):
        """Test initialize_system flow"""
        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({'driver': ['A', 'B']})
        mock_load_standings.return_value = pd.DataFrame({'driver': ['A', 'B']})
        mock_load_quali.return_value = pd.DataFrame({'driver': ['A', 'B']})
        mock_quali_features.return_value = pd.DataFrame({'driver': ['A', 'B']})
        mock_target.return_value = pd.DataFrame({
            'driver': ['A', 'B'],
            'promoted': [1, 0],
            'year': [2022, 2023]
        })

        features_df = pd.DataFrame({
            'driver': ['A', 'B'],
            'year': [2022, 2023],
            'feature1': [1, 2]
        })
        mock_engineer.return_value = features_df

        # Setup mock services
        mock_model_service = AsyncMock()
        mock_model_service_class.return_value = mock_model_service

        mock_prediction_service = AsyncMock()
        mock_pred_service_class.return_value = mock_prediction_service

        # Setup app_state mock
        data_service.app_state.save_state = Mock()

        await data_service.initialize_system()

        # Verify logging
        assert mock_logger.info.call_count >= 2  # Called for each series
        mock_logger.info.assert_any_call("Initializing system for f3_to_f2...")
        mock_logger.info.assert_any_call("Initializing system for f2_to_f1...")

        # Verify model training was called for both series
        assert mock_model_service.train_models.call_count == 2
        assert mock_model_service.save_models.call_count == 2

        # Verify predictions were updated
        assert mock_prediction_service.update_predictions.call_count == 2

        # Verify state was saved
        data_service.app_state.save_state.assert_called_once()

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.engineer_features')
    @patch('app.services.model_service.ModelService')
    @patch('app.services.prediction_service.PredictionService')
    @patch('app.services.data_service.LOGGER')
    async def test_initialize_system_no_historical_data(self, mock_logger, mock_pred_service_class,
                                                        mock_model_service_class, mock_engineer,
                                                        mock_target, mock_quali_features,
                                                        mock_load_quali, mock_load_standings,
                                                        mock_load_data, data_service):
        """Test warning when no historical data available for training"""
        # Setup mock data with only current year data
        mock_load_data.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_standings.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_quali.return_value = pd.DataFrame({'driver': ['A']})
        mock_quali_features.return_value = pd.DataFrame({'driver': ['A']})
        mock_target.return_value = pd.DataFrame({
            'driver': ['A'],
            'promoted': [1],
            'year': [CURRENT_YEAR]  # Only current year data
        })

        features_df = pd.DataFrame({
            'driver': ['A'],
            'year': [CURRENT_YEAR],
            'feature1': [1]
        })
        mock_engineer.return_value = features_df

        mock_prediction_service = AsyncMock()
        mock_pred_service_class.return_value = mock_prediction_service

        data_service.app_state.save_state = Mock()

        await data_service.initialize_system()

        # Verify warning was logged for no historical data
        mock_logger.warning.assert_any_call("No historical data available for training f3_to_f2")
        mock_logger.warning.assert_any_call("No historical data available for training f2_to_f1")

        # Verify ModelService was not called since no trainable data
        mock_model_service_class.assert_not_called()

    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.LOGGER')
    async def test_initialize_system_exception_handling(self, mock_logger, mock_load_data,
                                                        data_service):
        """Test exception handling in initialize_system"""
        # Make load_data raise an exception
        mock_load_data.side_effect = Exception("Test error")

        data_service.app_state.save_state = Mock()

        await data_service.initialize_system()

        # Verify errors were logged
        mock_logger.error.assert_any_call("Failed to initialize f3_to_f2: Test error")
        mock_logger.error.assert_any_call("Failed to initialize f2_to_f1: Test error")

        # Verify state was still saved despite errors
        data_service.app_state.save_state.assert_called_once()
