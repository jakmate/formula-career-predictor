import os
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.model_service import ModelService
from app.core.state import AppState
from app.core.predictor import RacingPredictor


@pytest.fixture
def mock_app_state():
    """Mock AppState with required attributes"""
    app_state = Mock(spec=AppState)
    app_state.models = {'f3_to_f2': {}, 'f2_to_f1': {}}
    app_state.scaler = {'f3_to_f2': Mock(), 'f2_to_f1': Mock()}
    app_state.feature_cols = {'f3_to_f2': ['col1', 'col2'], 'f2_to_f1': ['col1', 'col2']}
    app_state.system_status = {
        "models_available": {
            "f3_to_f2": [],
            "f2_to_f1": []
        },
        "last_training": None,
        "last_trained_season": None,
        "data_health": {}
    }
    return app_state


@pytest.fixture
def model_service(mock_app_state):
    """Create ModelService instance"""
    return ModelService(mock_app_state, series='f3_to_f2')


@pytest.fixture
def mock_trainable_df():
    """Mock training dataframe"""
    df = Mock()
    df.__len__ = Mock(return_value=100)
    df.__getitem__ = Mock(return_value=Mock(max=Mock(return_value=2023)))
    return df


class TestModelServiceInit:
    def test_init(self, mock_app_state):
        """Test initialization with series"""
        service = ModelService(mock_app_state, series='f3_to_f2')
        assert service.app_state == mock_app_state
        assert service.series == 'f3_to_f2'


class TestSaveModels:
    test_models_dir = os.path.join(os.sep, 'test', 'models')

    @patch('app.services.model_service.MODELS_DIR', test_models_dir)
    @patch('os.makedirs')
    @patch('torch.save')
    @patch('joblib.dump')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_save_models_with_series(self, mock_logger, mock_joblib_dump,
                                           mock_torch_save, mock_makedirs, model_service):
        """Test saving models for specific series"""
        # Setup mock models
        pytorch_model = Mock()
        pytorch_model.state_dict = Mock(return_value={'state': 'dict'})
        sklearn_model = Mock()
        model_service.app_state.models['f3_to_f2'] = {
            'PyTorch': pytorch_model,
            'RandomForest': sklearn_model
        }

        await model_service.save_models()

        # Verify directory creation (check that makedirs was called with series path)
        mock_makedirs.assert_called_once()
        call_args = mock_makedirs.call_args[0][0]  # Get the first positional argument
        assert call_args.endswith('f3_to_f2')  # Verify it ends with the series name

        # Verify PyTorch model save
        expected_path = os.path.join(os.sep, 'test', 'models', 'f3_to_f2', 'PyTorch.pt')
        mock_torch_save.assert_called_once_with(
            {'state': 'dict'},
            expected_path,
            _use_new_zipfile_serialization=True
        )

        # Verify sklearn model save and preprocessor save
        assert mock_joblib_dump.call_count == 2  # sklearn model + preprocessor

        mock_logger.info.assert_called_with("Models saved successfully for f3_to_f2")

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.makedirs')
    @patch('torch.save')
    @patch('joblib.dump')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_save_models_without_series(self, mock_logger, mock_joblib_dump,
                                              mock_torch_save, mock_makedirs, mock_app_state):
        """Test saving models without specific series"""
        service = ModelService(mock_app_state, series=None)
        mock_app_state.models = {'RandomForest': Mock()}
        mock_app_state.scaler = Mock()
        mock_app_state.feature_cols = ['col1', 'col2']

        await service.save_models()

        mock_makedirs.assert_called_once()
        call_args = mock_makedirs.call_args[0][0]
        # Should be base models directory when no series specified
        assert 'models' in str(call_args)
        mock_logger.info.assert_called_with("Models saved successfully for all series")

    @patch('os.makedirs')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_save_models_exception(self, mock_logger, mock_makedirs, model_service):
        """Test save models exception handling"""
        mock_makedirs.side_effect = Exception("Directory error")

        await model_service.save_models()

        mock_logger.error.assert_called_with("Error saving models: Directory error")


class TestLoadModels:
    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('joblib.load')
    @patch('torch.load')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_models_success(self, mock_logger, mock_torch_load,
                                       mock_joblib_load, mock_listdir, mock_exists,
                                       model_service):
        """Test successful model loading"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ['RandomForest.joblib', 'PyTorch.pt', 'preprocessor.joblib']

        # Mock preprocessor loading
        mock_joblib_load.side_effect = [
            {'scaler': Mock(), 'feature_cols': ['col1', 'col2']},  # preprocessor
            Mock()  # RandomForest model
        ]

        # Mock PyTorch loading
        mock_torch_load.return_value = {'param': 'value'}

        with patch.object(RacingPredictor, '__init__', return_value=None), \
             patch.object(RacingPredictor, 'load_state_dict'):

            result = await model_service.load_models()

            assert result is True
            mock_logger.info.assert_called()
            # Verify system status update
            assert len(model_service.app_state.system_status["models_available"]) > 0

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_models_no_directory(self, mock_logger, mock_exists, model_service):
        """Test loading when directory doesn't exist"""
        mock_exists.return_value = False

        result = await model_service.load_models()

        assert result is False

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('joblib.load')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_models_without_series(self, mock_logger, mock_joblib_load,
                                              mock_listdir, mock_exists, mock_app_state):
        """Test loading models for all series"""
        service = ModelService(mock_app_state, series=None)
        mock_exists.return_value = True
        mock_listdir.return_value = ['RandomForest.joblib', 'preprocessor.joblib']
        mock_joblib_load.side_effect = [
            {'scaler': Mock(), 'feature_cols': ['col1', 'col2']},
            Mock(),
            {'scaler': Mock(), 'feature_cols': ['col1', 'col2']},
            Mock()
        ]

        result = await service.load_models()

        assert result is True

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('joblib.load')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_load_models_exception(self, mock_logger, mock_joblib_load,
                                         mock_listdir, mock_exists, model_service):
        """Test load models exception handling"""
        mock_exists.return_value = True
        mock_listdir.side_effect = Exception("Directory read error")

        result = await model_service.load_models()

        assert result is False
        mock_logger.error.assert_called_with("Error loading models: Directory read error")

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('joblib.load')
    @pytest.mark.asyncio
    async def test_load_models_pytorch_loading(self, mock_joblib_load, mock_listdir,
                                               mock_exists, model_service):
        """Test PyTorch model loading specifically"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['PyTorch.pt', 'preprocessor.joblib']

        # Mock preprocessor
        mock_joblib_load.return_value = {
            'scaler': Mock(),
            'feature_cols': ['col1', 'col2']
        }

        with patch('torch.load') as mock_torch_load, \
            patch.object(RacingPredictor, '__init__', return_value=None), \
                patch.object(RacingPredictor, 'load_state_dict') as mock_load_state:

            mock_torch_load.return_value = {'param': 'value'}

            result = await model_service.load_models()

            assert result is True
            mock_torch_load.assert_called()
            mock_load_state.assert_called_once_with({'param': 'value'})


class TestTrainModels:
    @patch('app.core.predictor.train_models')
    @patch('app.services.model_service.LOGGER')
    @pytest.mark.asyncio
    async def test_train_models_success(self, mock_logger, mock_train_models,
                                        model_service, mock_trainable_df):
        """Test successful model training"""
        # Setup mock return values
        mock_models = {'RandomForest': Mock(), 'PyTorch': Mock()}
        mock_feature_cols = ['col1', 'col2', 'col3']
        mock_scaler = Mock()
        mock_train_models.return_value = (mock_models, mock_feature_cols, mock_scaler)

        await model_service.train_models(mock_trainable_df)

        # Verify training was called
        mock_train_models.assert_called_once_with(mock_trainable_df)

        # Verify state updates
        assert model_service.app_state.models['f3_to_f2'] == mock_models
        assert model_service.app_state.feature_cols['f3_to_f2'] == mock_feature_cols
        assert model_service.app_state.scaler['f3_to_f2'] == mock_scaler

        # Verify system status updates
        assert isinstance(model_service.app_state.system_status["last_training"], datetime)
        assert model_service.app_state.system_status["last_trained_season"] == 2023
        assert len(model_service.app_state.system_status["models_available"]["f3_to_f2"]) > 0

        # Verify data health update
        expected_health = {
            "historical_records": 100,
            "current_records": 0
        }
        assert model_service.app_state.system_status["data_health"]['f3_to_f2'] == expected_health

        mock_logger.info.assert_called_with(
            "Training classification models for f3_to_f2 on 100 records"
        )


class TestEdgeCases:
    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.makedirs')
    @patch('joblib.dump')
    @pytest.mark.asyncio
    async def test_save_models_empty_models_dict(self, mock_dump, mock_makedirs, model_service):
        """Test saving when models dict is empty"""
        model_service.app_state.models['f3_to_f2'] = {}

        await model_service.save_models()
        # Should still save preprocessor
        mock_dump.assert_called_once()

    @patch('app.services.model_service.MODELS_DIR', '/test/models')
    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('joblib.load')
    @pytest.mark.asyncio
    async def test_load_models_missing_preprocessor(self, mock_joblib_load,
                                                    mock_listdir, mock_exists,
                                                    model_service):
        """Test loading when preprocessor file is missing"""
        def exists_side_effect(path):
            return 'preprocessor.joblib' not in path

        mock_exists.side_effect = exists_side_effect
        mock_listdir.return_value = ['RandomForest.joblib']
        mock_joblib_load.return_value = Mock()

        result = await model_service.load_models()

        # Should still load models but may not set all state properly
        assert result is True

    @patch('app.core.predictor.train_models')
    @pytest.mark.asyncio
    async def test_train_models_empty_dataframe(self, mock_train_models, model_service):
        """Test training with empty dataframe"""
        empty_df = Mock()
        empty_df.__len__ = Mock(return_value=0)
        empty_df.__getitem__ = Mock(return_value=Mock(max=Mock(return_value=2023)))

        mock_train_models.return_value = ({}, [], Mock())

        await model_service.train_models(empty_df)

        # Verify it still processes even with 0 records
        health = model_service.app_state.system_status["data_health"]['f3_to_f2']
        assert health["historical_records"] == 0
        assert model_service.app_state.system_status["models_available"]["f3_to_f2"] == []
