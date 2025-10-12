import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np

from app.services.prediction_service import PredictionService
from app.models.predictions import PredictionsResponse, PredictionResponse
from app.core.state import AppState
from app.services.data_service import DataService


@pytest.fixture
def mock_app_state():
    """Create a mock AppState with necessary attributes"""
    app_state = Mock(spec=AppState)
    app_state.models = {
        'f1': {
            'RandomForest': Mock(),
            'PyTorch_MLP': Mock(),
            'Calibrated_SVM': Mock()
        }
    }
    app_state.feature_cols = {
        'f1': ['points', 'wins', 'podiums', 'age', 'experience']
    }
    app_state.scaler = {
        'f1': Mock()
    }
    app_state.system_status = {
        'status': 'active',
        'current_year': 2024,
        'last_update': datetime.now(),
        'last_scrape': datetime.now(),
        'last_training': datetime.now(),
        'models_available': ['RandomForest', 'PyTorch_MLP', 'Calibrated_SVM'],
        'data_health': {
            'status': {'code': 1},
            'last_check': {'timestamp': 1726844711}
        }
    }
    return app_state


@pytest.fixture
def mock_data_service():
    """Create a mock DataService"""
    return Mock(spec=DataService)


@pytest.fixture
def sample_dataframe():
    """Create sample current data DataFrame"""
    return pd.DataFrame({
        'Driver': ['Hamilton', 'Verstappen', 'Leclerc'],
        'nationality': ['British', 'Dutch', 'Monegasque'],
        'pos': [1, 2, 3],
        'points': [387.0, 365.0, 308.0],
        'avg_quali_pos': [2.1, 1.8, 3.2],
        'wins': [2, 7, 2],
        'win_rate': [0.09, 0.32, 0.09],
        'podiums': [9, 10, 5],
        'dnf_rate': [0.05, 0.05, 0.14],
        'experience': [16, 9, 6],
        'dob': ['1985-01-07', '1997-09-30', '1997-10-16'],
        'age': [39.0, 26.0, 26.0],
        'participation_rate': [1.0, 1.0, 1.0],
        'teammate_h2h_rate': [0.6, 0.8, 0.7],
        'team': ['Mercedes', 'Red Bull', 'Ferrari'],
        'team_pos': [3, 1, 2],
        'team_points': [409.0, 860.0, 406.0],
        'year': [2024, 2024, 2024]
    })


@pytest.fixture
def prediction_service(mock_app_state, mock_data_service):
    """Create PredictionService instance with proper dependencies"""
    return PredictionService(mock_app_state, 'f1', mock_data_service)


class TestPredictionServiceInitialization:
    def test_init_stores_app_state_and_series(self, mock_app_state, mock_data_service):
        service = PredictionService(mock_app_state, 'f1', mock_data_service)
        assert service.app_state == mock_app_state
        assert service.series == 'f1'
        assert service.data_service == mock_data_service


class TestGetPredictions:
    @pytest.mark.asyncio
    async def test_get_predictions_no_models_raises_error(self, mock_app_state, mock_data_service):
        mock_app_state.models = {'f1': {}}
        service = PredictionService(mock_app_state, 'f1', mock_data_service)

        with pytest.raises(ValueError, match="No models available for series f1"):
            await service.get_predictions()

    @pytest.mark.asyncio
    async def test_get_predictions_no_feature_cols_raises_error(
        self,
        mock_app_state,
        mock_data_service,
        sample_dataframe
    ):
        mock_app_state.feature_cols = {'f1': []}
        service = PredictionService(mock_app_state, 'f1', mock_data_service)

        with patch.object(service.data_service, 'load_current_data', return_value=sample_dataframe):
            with pytest.raises(ValueError, match="No feature columns available for series f1"):
                await service.get_predictions()

    @pytest.mark.asyncio
    async def test_get_predictions_model_error_handling(self, mock_app_state,
                                                        mock_data_service, sample_dataframe):
        service = PredictionService(mock_app_state, 'f1', mock_data_service)

        # Mock the data service
        with patch.object(service.data_service, 'load_current_data', return_value=sample_dataframe):
            # Mock _get_model_predictions to raise an error for one model
            with patch.object(service, '_get_model_predictions') as mock_get_predictions:
                mock_get_predictions.side_effect = [
                    np.array([0.8, 0.6, 0.4]),  # First model succeeds
                    Exception("Model error"),    # Second model fails
                    np.array([0.7, 0.5, 0.3])   # Third model succeeds
                ]

                with patch.object(service, '_create_prediction_responses') as mock_create_responses:
                    mock_create_responses.return_value = [
                        PredictionResponse(
                            driver="Hamilton", nationality="British", position=1,
                            points=387.0, avg_quali_pos=2.1, wins=2, win_rate=0.09,
                            podiums=9, dnf_rate=0.05, experience=16,
                            dob="1985-01-07", age=39.0, participation_rate=1.0,
                            teammate_h2h=0.6, team="Mercedes", team_pos=3,
                            team_points=409.0, empirical_percentage=80.0
                        )
                    ]

                    result = await service.get_predictions()

                    assert isinstance(result, PredictionsResponse)
                    assert len(result.predictions) == 3

                    # Check error model has empty predictions and numeric error count
                    error_model = result.predictions['PyTorch_MLP']
                    assert error_model.predictions == []
                    assert error_model.accuracy_metrics['error_count'] == 1


class TestGetModelPredictions:
    def test_get_model_predictions_model_not_found(self, prediction_service):
        X_current = pd.DataFrame({'points': [100], 'wins': [1]})

        with pytest.raises(ValueError, match="Model NonExistent not found for series f1"):
            prediction_service._get_model_predictions('NonExistent', X_current)

    def test_get_model_predictions_pytorch_model(self, prediction_service, mock_app_state):
        # Setup PyTorch model mock without calibrator
        pytorch_model = Mock()
        pytorch_model.eval = Mock()
        # Explicitly set calibrator to None to test the non-calibrated path
        pytorch_model.calibrator = None
        mock_app_state.models['f1']['PyTorch_MLP'] = pytorch_model

        # Setup scaler mock
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mock_app_state.scaler['f1'] = mock_scaler

        X_current = pd.DataFrame({
            'points': [100], 'wins': [1], 'podiums': [2], 'age': [25], 'experience': [5]
        })

        # Mock torch operations
        with patch('torch.no_grad'), \
             patch('torch.FloatTensor') as mock_tensor, \
             patch('torch.cuda.is_available', return_value=False):

            mock_tensor_instance = Mock()
            mock_tensor.return_value = mock_tensor_instance

            # Mock model output
            mock_logits = Mock()
            pytorch_model.return_value = mock_logits

            # Mock sigmoid output
            mock_sigmoid_output = Mock()
            mock_sigmoid_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.8]) # noqa: 501

            with patch('torch.sigmoid', return_value=mock_sigmoid_output):
                result = prediction_service._get_model_predictions('PyTorch_MLP', X_current)

                pytorch_model.eval.assert_called_once()
                mock_scaler.transform.assert_called_once()
                assert isinstance(result, np.ndarray)

    def test_get_model_predictions_pytorch_with_cuda(self, prediction_service, mock_app_state):
        # Setup PyTorch model mock without calibrator
        pytorch_model = Mock()
        pytorch_model.eval = Mock()
        pytorch_model.calibrator = None  # Test non-calibrated path
        mock_app_state.models['f1']['PyTorch_MLP'] = pytorch_model

        # Setup scaler mock
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mock_app_state.scaler['f1'] = mock_scaler

        X_current = pd.DataFrame({
            'points': [100], 'wins': [1], 'podiums': [2], 'age': [25], 'experience': [5]
        })

        # Mock torch operations with CUDA available
        with patch('torch.no_grad'), \
             patch('torch.FloatTensor') as mock_tensor, \
             patch('torch.cuda.is_available', return_value=True):

            mock_tensor_instance = Mock()
            mock_tensor_instance.cuda.return_value = mock_tensor_instance
            mock_tensor.return_value = mock_tensor_instance

            # Mock model output
            mock_logits = Mock()
            pytorch_model.return_value = mock_logits

            # Mock sigmoid output
            mock_sigmoid_output = Mock()
            mock_sigmoid_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.8]) # noqa: 501

            with patch('torch.sigmoid', return_value=mock_sigmoid_output):
                result = prediction_service._get_model_predictions('PyTorch_MLP', X_current)

                mock_tensor_instance.cuda.assert_called_once()
                assert isinstance(result, np.ndarray)

    def test_get_model_predictions_sklearn_model(self, prediction_service, mock_app_state):
        # Setup sklearn model mock without calibrator
        sklearn_model = Mock()
        sklearn_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.4, 0.6]])
        sklearn_model.calibrator = None  # Test non-calibrated path
        mock_app_state.models['f1']['RandomForest'] = sklearn_model

        X_current = pd.DataFrame({
            'points': [100, 200], 'wins': [1, 2], 'podiums': [2, 4],
            'age': [25, 30], 'experience': [5, 10]
        })

        result = prediction_service._get_model_predictions('RandomForest', X_current)

        sklearn_model.predict_proba.assert_called_once_with(X_current)
        np.testing.assert_array_equal(result, np.array([0.8, 0.6]))

    def test_get_model_predictions_with_calibrator(self, prediction_service, mock_app_state):
        # Setup model with calibrator
        model_with_calibrator = Mock()
        model_with_calibrator.predict_proba.return_value = np.array([[0.2, 0.8]])

        # Setup calibrator
        calibrator = Mock()
        calibrator.transform.return_value = np.array([0.75])
        model_with_calibrator.calibrator = calibrator

        mock_app_state.models['f1']['Calibrated_SVM'] = model_with_calibrator

        X_current = pd.DataFrame({
            'points': [100], 'wins': [1], 'podiums': [2],
            'age': [25], 'experience': [5]
        })

        result = prediction_service._get_model_predictions('Calibrated_SVM', X_current)

        calibrator.transform.assert_called_once_with(np.array([0.8]))
        np.testing.assert_array_equal(result, np.array([0.75]))


class TestCreatePredictionResponses:
    def test_create_prediction_responses(self, prediction_service, sample_dataframe):
        calibrated_probas = np.array([0.8, 0.6, 0.4])

        result = prediction_service._create_prediction_responses(
            sample_dataframe, calibrated_probas
        )

        assert len(result) == 3
        assert all(isinstance(pred, PredictionResponse) for pred in result)

        # Check sorting by empirical_percentage (descending)
        assert result[0].empirical_percentage == 80.0  # Hamilton
        assert result[1].empirical_percentage == 60.0  # Verstappen
        assert result[2].empirical_percentage == 40.0  # Leclerc

        # Check first prediction details
        first_pred = result[0]
        assert first_pred.driver == 'Hamilton'
        assert first_pred.nationality == 'British'
        assert first_pred.position == 1
        assert first_pred.points == 387.0
        assert first_pred.team == 'Mercedes'

    def test_create_prediction_responses_missing_optional_fields(self, prediction_service):
        # DataFrame with missing optional fields
        df_minimal = pd.DataFrame({
            'Driver': ['TestDriver'],
            'pos': [1],
            'points': [100.0],
            'wins': [1],
            'win_rate': [0.1],
            'podiums': [2],
            'dnf_rate': [0.1],
            'experience': [5],
            'participation_rate': [1.0],
            'teammate_h2h_rate': [0.6],
            'team': ['TestTeam'],
            'team_pos': [1],
            'team_points': [200.0]
        })

        calibrated_probas = np.array([0.5])

        result = prediction_service._create_prediction_responses(
            df_minimal, calibrated_probas
        )

        assert len(result) == 1
        pred = result[0]
        assert pred.driver == 'TestDriver'
        assert pred.nationality is None
        assert pred.dob is None
        assert pred.age is None


class TestUpdatePredictions:
    @pytest.mark.asyncio
    async def test_update_predictions_no_features_df(self, prediction_service, sample_dataframe):
        with patch.object(prediction_service.data_service, 'load_current_data', return_value=sample_dataframe): # noqa: 501
            with patch.object(prediction_service, '_get_model_predictions') as mock_get_predictions:
                mock_get_predictions.return_value = np.array([0.8, 0.6, 0.4])

                await prediction_service.update_predictions()

                # Check that current_predictions was created and populated
                assert hasattr(prediction_service.app_state, 'current_predictions')
                assert 'f1' in prediction_service.app_state.current_predictions
                assert len(prediction_service.app_state.current_predictions['f1']) == 3

    @pytest.mark.asyncio
    async def test_update_predictions_with_features_df(self, prediction_service, sample_dataframe):
        with patch.object(prediction_service, '_get_model_predictions') as mock_get_predictions:
            mock_get_predictions.return_value = np.array([0.8, 0.6, 0.4])

            await prediction_service.update_predictions(features_df=sample_dataframe)

            # Check that predictions were generated
            assert hasattr(prediction_service.app_state, 'current_predictions')
            assert 'f1' in prediction_service.app_state.current_predictions

    @pytest.mark.asyncio
    async def test_update_predictions_empty_dataframe(self, prediction_service):
        empty_df = pd.DataFrame()

        with patch.object(prediction_service.data_service, 'load_current_data', return_value=empty_df): # noqa: 501
            await prediction_service.update_predictions()

            # Should not create predictions for empty dataframe
            if hasattr(prediction_service.app_state, 'current_predictions'):
                assert 'f1' not in prediction_service.app_state.current_predictions

    @pytest.mark.asyncio
    async def test_update_predictions_model_error_handling(self, prediction_service, sample_dataframe): # noqa: 501
        with patch.object(prediction_service.data_service, 'load_current_data', return_value=sample_dataframe): # noqa: 501
            with patch.object(prediction_service, '_get_model_predictions') as mock_get_predictions:
                # First model succeeds, second fails, third succeeds
                mock_get_predictions.side_effect = [
                    np.array([0.8, 0.6, 0.4]),  # RandomForest succeeds
                    Exception("PyTorch model error"),  # PyTorch fails
                    np.array([0.7, 0.5, 0.3])   # Calibrated_SVM succeeds
                ]

                await prediction_service.update_predictions()

                # Should still create predictions for successful models
                assert hasattr(prediction_service.app_state, 'current_predictions')
                assert 'f1' in prediction_service.app_state.current_predictions
                # Only 2 successful predictions (failed one is skipped)
                assert len(prediction_service.app_state.current_predictions['f1']) == 2

    @pytest.mark.asyncio
    async def test_update_predictions_filters_by_current_year(self, prediction_service):
        # DataFrame with mixed years
        mixed_year_df = pd.DataFrame({
            'Driver': ['Hamilton', 'Verstappen', 'OldDriver'],
            'year': [2024, 2024, 2020],  # One old entry
            'points': [100, 200, 50],
            'wins': [1, 2, 0],
            'podiums': [2, 4, 1],
            'age': [25, 30, 35],
            'experience': [5, 10, 15]
        })

        with patch.object(prediction_service, '_get_model_predictions') as mock_get_predictions:
            mock_get_predictions.return_value = np.array([0.8, 0.6])  # Only 2 current year drivers

            await prediction_service.update_predictions(features_df=mixed_year_df)

            # Should filter to only current year (2024) entries
            mock_get_predictions.assert_called()

            # Verify predictions were created
            assert hasattr(prediction_service.app_state, 'current_predictions')
            assert 'f1' in prediction_service.app_state.current_predictions

    @pytest.mark.asyncio
    async def test_update_predictions_general_exception_handling(self, prediction_service):
        with patch.object(prediction_service.data_service, 'load_current_data', side_effect=Exception("Data loading error")): # noqa: 501
            # Should not raise exception, just log error
            await prediction_service.update_predictions()

            # Should not create predictions when there's a general error
            if hasattr(prediction_service.app_state, 'current_predictions'):
                assert 'f1' not in getattr(prediction_service.app_state, 'current_predictions', {})

    @pytest.mark.asyncio
    @patch('app.services.prediction_service.PredictionService._create_prediction_responses')
    @patch('app.services.prediction_service.PredictionService._get_model_predictions')
    async def test_get_predictions_cache_hit_scenario(self,
                                                      mock_get_predictions,
                                                      mock_create_responses,
                                                      prediction_service,
                                                      sample_dataframe):
        """Test cache hit scenario in get_predictions - covers cache functionality"""
        # Setup cache with existing data
        cache_key = "f1_processed_features"
        X_current = sample_dataframe[['points', 'wins', 'podiums', 'age', 'experience']].fillna(0)

        prediction_service.prediction_cache[cache_key] = {
            'current_df': sample_dataframe,
            'X_current': X_current,
            'timestamp': datetime.now()
        }

        # patch instance data_service.load_current_data at runtime
        with patch.object(prediction_service.data_service, 'load_current_data') as mock_load_data:
            # Configure mocks (these should not be called for cache hit, but set returns to be safe)
            mock_load_data.return_value = None
            mock_get_predictions.return_value = np.array([0.8, 0.6, 0.4])
            mock_create_responses.return_value = [
                PredictionResponse(
                    driver="Hamilton", nationality="British", position=1,
                    points=387.0, avg_quali_pos=2.1, wins=2, win_rate=0.09,
                    podiums=9, dnf_rate=0.05, experience=16,
                    dob="1985-01-07", age=39.0, participation_rate=1.0,
                    teammate_h2h=0.6, team="Mercedes", team_pos=3,
                    team_points=409.0, empirical_percentage=80.0
                )
            ]

            result = await prediction_service.get_predictions()

            # Verify cache was used (data_service.load_current_data should not be called)
            mock_load_data.assert_not_called()

            # Verify result structure
            assert isinstance(result, PredictionsResponse)
            assert len(result.predictions) == 3

    @patch('app.services.prediction_service.LOGGER')
    def test_clear_prediction_cache(self, mock_logger, prediction_service):
        """Test clear_prediction_cache method"""
        # Add some data to cache
        prediction_service.prediction_cache = {
            'f1_processed_features': {'data': 'test'},
            'f2_processed_features': {'data': 'test2'}
        }

        prediction_service.clear_prediction_cache()

        # Verify cache is cleared
        assert prediction_service.prediction_cache == {}

        # Verify logging
        mock_logger.info.assert_called_once_with("Cleared prediction cache for f1")

    @pytest.mark.asyncio
    @patch('app.services.prediction_service.LOGGER')
    async def test_update_predictions_exception_in_try_block(self, mock_logger, prediction_service):
        """Test exception handling in update_predictions"""
        # patch the instance method to raise
        with patch.object(prediction_service.data_service, 'load_current_data',
                          side_effect=Exception("Test error in update_predictions")):
            # This should not raise because update_predictions catches exceptions
            await prediction_service.update_predictions()

        # Verify error was logged
        mock_logger.error.assert_called_once_with(
            "Prediction update failed for f1: Test error in update_predictions"
        )

    @pytest.mark.asyncio
    @patch('app.services.prediction_service.PredictionService._get_model_predictions')
    async def test_update_predictions_with_empty_current_predictions(self,
                                                                     mock_get_predictions,
                                                                     prediction_service,
                                                                     sample_dataframe):
        """Test update_predictions when current_predictions attribute doesn't exist"""
        # Ensure current_predictions attribute doesn't exist
        if hasattr(prediction_service.app_state, 'current_predictions'):
            delattr(prediction_service.app_state, 'current_predictions')

        mock_get_predictions.return_value = np.array([0.8, 0.6, 0.4])

        # patch instance load_current_data to return the sample dataframe
        with patch.object(prediction_service.data_service, 'load_current_data', return_value=sample_dataframe): # noqa: 501
            await prediction_service.update_predictions()

        # Verify current_predictions was created
        assert hasattr(prediction_service.app_state, 'current_predictions')
        assert 'f1' in prediction_service.app_state.current_predictions
