import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from app.services.prediction_service import PredictionService
from app.models.predictions import PredictionsResponse, ModelResults, PredictionResponse
from app.core.state import AppState
from app.services.data_service import DataService


@pytest.fixture
def mock_app_state():
    """Create a mock AppState with necessary attributes"""
    state = Mock(spec=AppState)
    state.models = {
        'f1': {
            'RandomForest': Mock(),
            'PyTorch_MLP': Mock()
        }
    }
    state.feature_cols = {
        'f1': ['points', 'wins', 'podiums', 'dnf_rate', 'experience']
    }
    state.scaler = {
        'f1': Mock()
    }
    state.system_status = {
        'last_scrape': datetime(2024, 10, 1, 12, 0, 0),
        'last_training': datetime(2024, 10, 1, 10, 0, 0),
        'models_available': {'f1': ['RandomForest', 'PyTorch_MLP']},
        'data_health': {'f1': {'records': 1000, 'missing': 0}}
    }
    state.current_predictions = {}
    return state


@pytest.fixture
def mock_data_service():
    """Create a mock DataService"""
    return Mock(spec=DataService)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'Driver': ['Hamilton', 'Verstappen', 'Leclerc'],
        'nationality': ['British', 'Dutch', 'Monegasque'],
        'pos': [1, 2, 3],
        'points': [400.0, 380.0, 350.0],
        'avg_quali_pos': [2.5, 1.8, 3.2],
        'wins': [10, 12, 8],
        'win_rate': [0.45, 0.52, 0.35],
        'podiums': [15, 18, 14],
        'dnf_rate': [0.05, 0.03, 0.08],
        'experience': [280, 160, 100],
        'dob': ['1985-01-07', '1997-09-30', '1997-10-16'],
        'age': [39.0, 27.0, 27.0],
        'participation_rate': [0.98, 0.99, 0.97],
        'teammate_h2h_rate': [0.65, 0.70, 0.55],
        'team': ['Mercedes', 'Red Bull', 'Ferrari'],
        'team_pos': [1, 1, 1],
        'team_points': [600.0, 650.0, 580.0],
        'year': [2024, 2024, 2024]
    })


@pytest.fixture
def prediction_service(mock_app_state, mock_data_service):
    """Create PredictionService instance"""
    return PredictionService(
        app_state=mock_app_state,
        series='f1',
        data_service=mock_data_service
    )


class TestPredictionServiceInit:
    """Test initialization of PredictionService"""
    def test_init_sets_attributes(self, mock_app_state, mock_data_service):
        service = PredictionService(mock_app_state, 'f1', mock_data_service)
        assert service.app_state == mock_app_state
        assert service.series == 'f1'
        assert service.data_service == mock_data_service
        assert service.prediction_cache == {}


class TestGetPredictions:
    """Test get_predictions method"""
    @pytest.mark.asyncio
    async def test_get_predictions_success(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        # Setup
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        # Mock model predictions
        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_rf_model.calibrator = None

        mock_pytorch_model = prediction_service.app_state.models['f1']['PyTorch_MLP']
        mock_pytorch_model.eval = Mock()
        mock_pytorch_model.calibrator = None

        # Mock scaler
        prediction_service.app_state.scaler['f1'].transform.return_value = np.random.rand(3, 5)

        # Mock PyTorch forward pass
        with patch('torch.no_grad'), \
             patch('torch.FloatTensor'), \
             patch('torch.cuda.is_available', return_value=False):

            mock_output = Mock()
            mock_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.65, 0.55, 0.45]) # noqa: 501
            mock_pytorch_model.return_value = mock_output

            with patch('torch.sigmoid', return_value=mock_output):
                result = await prediction_service.get_predictions()

        # Assertions
        assert isinstance(result, PredictionsResponse)
        assert len(result.models) == 2
        assert 'RandomForest' in result.models
        assert 'PyTorch_MLP' in result.models
        assert isinstance(result.predictions['RandomForest'], ModelResults)
        assert len(result.predictions['RandomForest'].predictions) == 3

    @pytest.mark.asyncio
    async def test_get_predictions_no_models_raises_error(
        self, prediction_service
    ):
        prediction_service.app_state.models['f1'] = {}

        with pytest.raises(ValueError, match="No models available for series f1"):
            await prediction_service.get_predictions()

    @pytest.mark.asyncio
    async def test_get_predictions_no_feature_cols_raises_error(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)
        prediction_service.app_state.feature_cols['f1'] = []

        with pytest.raises(ValueError, match="No feature columns available"):
            await prediction_service.get_predictions()

    @pytest.mark.asyncio
    async def test_get_predictions_uses_cache(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        # First call - should cache
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_rf_model.calibrator = None

        prediction_service.app_state.models['f1'] = {'RandomForest': mock_rf_model}

        await prediction_service.get_predictions()
        first_call_count = mock_data_service.load_current_data.call_count

        # Second call - should use cache
        await prediction_service.get_predictions()
        second_call_count = mock_data_service.load_current_data.call_count

        assert first_call_count == 1
        assert second_call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_get_predictions_handles_model_error(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.side_effect = Exception("Model error")

        result = await prediction_service.get_predictions()

        assert isinstance(result, PredictionsResponse)
        assert result.predictions['RandomForest'].predictions == []
        assert result.predictions['RandomForest'].accuracy_metrics['error_count'] == 1


class TestGetModelPredictions:
    """Test _get_model_predictions method"""
    def test_sklearn_model_predictions(self, prediction_service):
        X_current = pd.DataFrame(np.random.rand(3, 5))
        mock_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_model.calibrator = None

        result = prediction_service._get_model_predictions('RandomForest', X_current)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_equal(result, np.array([0.7, 0.6, 0.5]))

    def test_pytorch_model_predictions(self, prediction_service):
        X_current = pd.DataFrame(np.random.rand(3, 5))
        mock_model = prediction_service.app_state.models['f1']['PyTorch_MLP']
        mock_model.eval = Mock()
        mock_model.calibrator = None

        prediction_service.app_state.scaler['f1'].transform.return_value = np.random.rand(3, 5)

        with patch('torch.no_grad'), \
             patch('torch.FloatTensor'), \
             patch('torch.cuda.is_available', return_value=False):

            mock_output = Mock()
            mock_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.65, 0.55, 0.45]) # noqa: 501
            mock_model.return_value = mock_output

            with patch('torch.sigmoid', return_value=mock_output):
                result = prediction_service._get_model_predictions('PyTorch_MLP', X_current)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_model_with_calibrator(self, prediction_service):
        X_current = pd.DataFrame(np.random.rand(3, 5))
        mock_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])

        mock_calibrator = Mock()
        mock_calibrator.transform.return_value = np.array([0.75, 0.65, 0.55])
        mock_model.calibrator = mock_calibrator

        result = prediction_service._get_model_predictions('RandomForest', X_current)

        np.testing.assert_array_equal(result, np.array([0.75, 0.65, 0.55]))
        mock_calibrator.transform.assert_called_once()

    def test_model_not_found_raises_error(self, prediction_service):
        X_current = pd.DataFrame(np.random.rand(3, 5))

        with pytest.raises(ValueError, match="Model InvalidModel not found"):
            prediction_service._get_model_predictions('InvalidModel', X_current)


class TestCreatePredictionResponses:
    """Test _create_prediction_responses method"""
    def test_creates_prediction_responses(self, prediction_service, sample_dataframe):
        calibrated_probas = np.array([0.75, 0.65, 0.55])

        result = prediction_service._create_prediction_responses(
            sample_dataframe, calibrated_probas
        )

        assert len(result) == 3
        assert all(isinstance(pred, PredictionResponse) for pred in result)
        assert result[0].driver == 'Hamilton'
        assert pytest.approx(result[0].empirical_percentage, rel=1e-9) == 75.0
        assert pytest.approx(result[1].empirical_percentage, rel=1e-9) == 65.0
        assert pytest.approx(result[2].empirical_percentage, rel=1e-9) == 55.0

    def test_predictions_sorted_by_percentage(self, prediction_service, sample_dataframe):
        calibrated_probas = np.array([0.55, 0.75, 0.65])  # Unsorted

        result = prediction_service._create_prediction_responses(
            sample_dataframe, calibrated_probas
        )

        # Should be sorted descending
        assert pytest.approx(result[0].empirical_percentage, rel=1e-9) == 75.0
        assert pytest.approx(result[1].empirical_percentage, rel=1e-9) == 65.0
        assert pytest.approx(result[2].empirical_percentage, rel=1e-9) == 55.0


class TestUpdatePredictions:
    """Test update_predictions method"""
    @pytest.mark.asyncio
    async def test_update_predictions_success(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_rf_model.calibrator = None

        # Also mock PyTorch model to match the actual behavior
        mock_pytorch_model = prediction_service.app_state.models['f1']['PyTorch_MLP']
        mock_pytorch_model.eval = Mock()
        mock_pytorch_model.calibrator = None
        prediction_service.app_state.scaler['f1'].transform.return_value = np.random.rand(3, 5)

        with patch('torch.no_grad'), \
             patch('torch.FloatTensor'), \
             patch('torch.cuda.is_available', return_value=False):

            mock_output = Mock()
            mock_output.cpu.return_value.numpy.return_value.flatten.return_value = np.array([0.65, 0.55, 0.45]) # noqa: 501
            mock_pytorch_model.return_value = mock_output

            with patch('torch.sigmoid', return_value=mock_output):
                await prediction_service.update_predictions()

        assert 'f1' in prediction_service.app_state.current_predictions
        assert len(prediction_service.app_state.current_predictions['f1']) == 2

    @pytest.mark.asyncio
    async def test_update_predictions_with_features_df(
        self, prediction_service, sample_dataframe
    ):
        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_rf_model.calibrator = None

        await prediction_service.update_predictions(features_df=sample_dataframe)

        assert 'f1' in prediction_service.app_state.current_predictions

    @pytest.mark.asyncio
    async def test_update_predictions_empty_dataframe(
        self, prediction_service, mock_data_service
    ):
        mock_data_service.load_current_data = AsyncMock(return_value=pd.DataFrame())

        await prediction_service.update_predictions()

        # Should not raise error, just log warning
        assert 'f1' not in prediction_service.app_state.current_predictions or \
               prediction_service.app_state.current_predictions.get('f1') is None

    @pytest.mark.asyncio
    async def test_update_predictions_model_failure(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.side_effect = Exception("Prediction error")

        # Should not raise, just log error
        await prediction_service.update_predictions()


class TestClearPredictionCache:
    """Test clear_prediction_cache method"""
    def test_clears_cache(self, prediction_service):
        # Add some cache data
        prediction_service.prediction_cache['test_key'] = {'data': 'value'}

        prediction_service.clear_prediction_cache()

        assert len(prediction_service.prediction_cache) == 0


class TestIntegration:
    """Integration tests combining multiple methods"""
    @pytest.mark.asyncio
    async def test_full_prediction_workflow(
        self, prediction_service, mock_data_service, sample_dataframe
    ):
        """Test complete workflow from data loading to predictions"""
        mock_data_service.load_current_data = AsyncMock(return_value=sample_dataframe)

        # Setup models
        mock_rf_model = prediction_service.app_state.models['f1']['RandomForest']
        mock_rf_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5]])
        mock_rf_model.calibrator = None

        # Remove PyTorch model for simpler test
        prediction_service.app_state.models['f1'] = {'RandomForest': mock_rf_model}
        # Also update system_status to reflect single model
        prediction_service.app_state.system_status['models_available'] = {'f1': ['RandomForest']}

        # Get predictions
        result = await prediction_service.get_predictions()

        # Verify complete response structure
        assert isinstance(result, PredictionsResponse)
        assert len(result.predictions) == 1
        assert result.predictions['RandomForest'].predictions[0].driver == 'Hamilton'
        assert result.system_status.models_available == {'f1': ['RandomForest']}

        # Clear cache and verify
        prediction_service.clear_prediction_cache()
        assert len(prediction_service.prediction_cache) == 0
