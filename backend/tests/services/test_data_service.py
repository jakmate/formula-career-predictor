import pytest
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from fastapi import HTTPException

from app.services.data_service import DataService
from app.core.state import AppState
from app.config import CURRENT_YEAR


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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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
    @pytest.mark.asyncio
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

    @patch('app.services.data_service.time.time')
    @patch('app.services.data_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_current_data_cache_hit(self, mock_logger, mock_time, data_service):
        """Test cache hit scenario - covers lines 20-21"""
        # Setup cached data
        cache_key = "current_data_f2_to_f1"
        cached_data = pd.DataFrame({'driver': ['A'], 'year': [2023]})
        data_service.data_cache[cache_key] = cached_data

        # Mock time to verify performance logging
        mock_time.side_effect = [1000, 1000.5]  # start and end times

        result = await data_service.load_current_data('f2_to_f1')

        # Verify cache hit was logged
        mock_logger.info.assert_called_once_with(
            "Cache HIT for f2_to_f1 - returned in 0.50s"
        )

        # Verify cached data was returned
        pd.testing.assert_frame_equal(result, cached_data)

    @patch('app.core.predictor.engineer_features')
    @patch('app.core.predictor.create_target_variable')
    @patch('app.core.predictor.calculate_qualifying_features')
    @patch('app.services.data_service.load_qualifying_data')
    @patch('app.services.data_service.load_standings_data')
    @patch('app.services.data_service.load_data')
    @patch('app.services.data_service.time.time')
    @patch('app.services.data_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_current_data_cache_miss(self, mock_logger, mock_time,
                                                mock_load_data, mock_load_standings,
                                                mock_load_quali, mock_quali_features,
                                                mock_target, mock_engineer, data_service):
        """Test cache miss scenario"""
        # Ensure cache is empty
        data_service.data_cache.clear()

        # Setup mock data
        mock_load_data.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_standings.return_value = pd.DataFrame({'driver': ['A']})
        mock_load_quali.return_value = pd.DataFrame({'driver': ['A']})
        mock_quali_features.return_value = pd.DataFrame({'driver': ['A']})
        mock_target.return_value = pd.DataFrame({
            'driver': ['A'],
            'promoted': [1]
        })

        # Mock features with current year data
        features_df = pd.DataFrame({
            'driver': ['A'],
            'year': [CURRENT_YEAR],
            'feature1': [1]
        })
        mock_engineer.return_value = features_df

        # Use a counter to return incremental time values
        time_counter = 1000.0

        def time_side_effect():
            nonlocal time_counter
            current_time = time_counter
            time_counter += 0.1
            return current_time

        mock_time.side_effect = time_side_effect

        result = await data_service.load_current_data('f2_to_f1')

        # Verify cache miss was logged
        mock_logger.info.assert_any_call(
            "Cache MISS for f2_to_f1 - processing data..."
        )

        # Verify total processing was logged (check the pattern, not exact string)
        total_processing_calls = [
            c for c in mock_logger.info.call_args_list
            if c[0][0].startswith("Total processing for f2_to_f1:")
        ]
        assert len(total_processing_calls) == 1
        assert "cached 1 records" in total_processing_calls[0][0][0]

        # Verify data was cached
        cache_key = "current_data_f2_to_f1"
        assert cache_key in data_service.data_cache
        pd.testing.assert_frame_equal(
            data_service.data_cache[cache_key], result
        )

    @patch('app.services.data_service.LOGGER')
    def test_clear_cache_specific_series(self, mock_logger, data_service):
        """Test clearing cache for specific series"""
        # Setup cache with multiple entries
        data_service.data_cache = {
            "current_data_f2_to_f1": pd.DataFrame({'a': [1]}),
            "current_data_f3_to_f2": pd.DataFrame({'b': [2]}),
            "full_data_f2_to_f1": pd.DataFrame({'c': [3]}),
            "other_data": pd.DataFrame({'d': [4]})
        }

        data_service.clear_cache('f2_to_f1')

        # Verify only f2_to_f1 related entries were removed
        assert "current_data_f2_to_f1" not in data_service.data_cache
        assert "full_data_f2_to_f1" not in data_service.data_cache
        assert "current_data_f3_to_f2" in data_service.data_cache
        assert "other_data" in data_service.data_cache

        # Verify logging
        mock_logger.info.assert_called_once_with("Cleared cache for f2_to_f1")

    @patch('app.services.data_service.LOGGER')
    def test_clear_cache_all(self, mock_logger, data_service):
        """Test clearing entire cache"""
        # Setup cache
        data_service.data_cache = {
            "current_data_f2_to_f1": pd.DataFrame({'a': [1]}),
            "current_data_f3_to_f2": pd.DataFrame({'b': [2]})
        }

        data_service.clear_cache()

        # Verify cache is empty
        assert data_service.data_cache == {}

        # Verify logging
        mock_logger.info.assert_called_once_with("Cleared all cached data")

    @patch('app.services.data_service.LOGGER')
    def test_clear_cache_nonexistent_series(self, mock_logger, data_service):
        """Test clearing cache for series that doesn't exist in cache"""
        # Setup cache with different series
        data_service.data_cache = {
            "current_data_f3_to_f2": pd.DataFrame({'b': [2]})
        }

        data_service.clear_cache('f2_to_f1')

        # Verify cache unchanged
        assert "current_data_f3_to_f2" in data_service.data_cache
        assert len(data_service.data_cache) == 1

        # Verify logging still occurred
        mock_logger.info.assert_called_once_with("Cleared cache for f2_to_f1")
