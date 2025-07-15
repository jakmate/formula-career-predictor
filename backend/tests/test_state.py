import pytest
import json
from datetime import datetime
from unittest.mock import patch, mock_open

from app.core.state import AppState


@pytest.fixture
def mock_state_file():
    with patch('app.core.state.STATE_FILE', '/tmp/test_state.json'):
        yield '/tmp/test_state.json'


@pytest.fixture
def sample_state_data():
    return {
        "last_scrape": "2024-01-01T12:00:00",
        "last_training": "2024-01-01T13:00:00",
        "last_trained_season": "2024",
        "models_available": ["model1", "model2"]
    }


class TestAppStateInit:
    def test_init_default_values(self):
        state = AppState()

        assert state.models == {}
        assert state.feature_cols == []
        assert state.scaler is None
        assert state.current_predictions == []
        assert state.system_status["last_scrape"] is None
        assert state.system_status["last_training"] is None
        assert state.system_status["last_trained_season"] is None
        assert state.system_status["models_available"] == []
        assert state.system_status["data_health"] == {}
        assert state.scheduler is not None


class TestSaveState:
    def test_save_state_with_datetime_values(self, mock_state_file):
        state = AppState()
        test_time = datetime(2024, 1, 1, 12, 0, 0)

        state.system_status["last_scrape"] = test_time
        state.system_status["last_training"] = test_time
        state.system_status["last_trained_season"] = "2024"
        state.system_status["models_available"] = ["model1"]

        with patch('builtins.open', mock_open()) as mock_file:
            state.save_state()

            mock_file.assert_called_once_with(mock_state_file, 'w')

            # Check JSON was written
            handle = mock_file.return_value.__enter__.return_value
            written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
            saved_data = json.loads(written_data)

            assert saved_data["last_scrape"] == "2024-01-01T12:00:00"
            assert saved_data["last_training"] == "2024-01-01T12:00:00"
            assert saved_data["last_trained_season"] == "2024"
            assert saved_data["models_available"] == ["model1"]

    def test_save_state_with_none_values(self):
        state = AppState()

        with patch('builtins.open', mock_open()) as mock_file:
            state.save_state()

            handle = mock_file.return_value.__enter__.return_value
            written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
            saved_data = json.loads(written_data)

            assert saved_data["last_scrape"] is None
            assert saved_data["last_training"] is None
            assert saved_data["last_trained_season"] is None
            assert saved_data["models_available"] == []


class TestLoadState:
    def test_load_state_success(self, sample_state_data):
        state = AppState()

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(sample_state_data))):

            result = state.load_state()

            assert result is True
            assert state.system_status["last_scrape"] == datetime(2024, 1, 1, 12, 0, 0)
            assert state.system_status["last_training"] == datetime(2024, 1, 1, 13, 0, 0)
            assert state.system_status["last_trained_season"] == "2024"
            assert state.system_status["models_available"] == ["model1", "model2"]

    def test_load_state_file_not_exists(self):
        state = AppState()

        with patch('os.path.exists', return_value=False):
            result = state.load_state()

            assert result is False
            # State should remain at default values
            assert state.system_status["last_scrape"] is None

    def test_load_state_none_datetime_values(self):
        state = AppState()
        state_data = {
            "last_scrape": None,
            "last_training": None,
            "last_trained_season": "2024",
            "models_available": ["model1"]
        }

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(state_data))):

            result = state.load_state()

            assert result is True
            assert state.system_status["last_scrape"] is None
            assert state.system_status["last_training"] is None
            assert state.system_status["last_trained_season"] == "2024"

    def test_load_state_json_decode_error(self, mock_state_file):
        state = AppState()

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid json")), \
             patch('app.core.state.LOGGER') as mock_logger, \
             patch('os.rename') as mock_rename:

            result = state.load_state()

            assert result is False
            mock_logger.error.assert_called()
            mock_rename.assert_called_once_with(mock_state_file, f"{mock_state_file}.backup")

    def test_load_state_general_exception(self):
        state = AppState()

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', side_effect=IOError("File error")), \
             patch('app.core.state.LOGGER') as mock_logger:

            result = state.load_state()

            assert result is False
            mock_logger.error.assert_called()

    def test_load_state_datetime_parsing(self):
        state = AppState()
        state_data = {
            "last_scrape": "2024-06-15T14:30:45",
            "last_training": "2024-06-15T15:45:30",
            "last_trained_season": "2024",
            "models_available": []
        }

        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(state_data))):

            result = state.load_state()

            assert result is True
            assert state.system_status["last_scrape"] == datetime(2024, 6, 15, 14, 30, 45)
            assert state.system_status["last_training"] == datetime(2024, 6, 15, 15, 45, 30)


class TestStateIntegration:
    """Test save/load integration"""

    def test_save_load_roundtrip(self):
        """Test saving and loading state maintains data integrity"""
        state1 = AppState()
        test_time = datetime(2024, 1, 1, 12, 0, 0)

        # Set up state
        state1.system_status["last_scrape"] = test_time
        state1.system_status["last_training"] = test_time
        state1.system_status["last_trained_season"] = "2024"
        state1.system_status["models_available"] = ["model1", "model2"]

        # Save state
        saved_data = None
        with patch('builtins.open', mock_open()) as mock_file:
            state1.save_state()
            handle = mock_file.return_value.__enter__.return_value
            saved_data = ''.join(call.args[0] for call in handle.write.call_args_list)

        # Load into new state object
        state2 = AppState()
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=saved_data)):

            result = state2.load_state()

            assert result is True
            assert state2.system_status["last_scrape"] == test_time
            assert state2.system_status["last_training"] == test_time
            assert state2.system_status["last_trained_season"] == "2024"
            assert state2.system_status["models_available"] == ["model1", "model2"]
