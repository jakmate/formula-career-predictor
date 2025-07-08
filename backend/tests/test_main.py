from datetime import datetime
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app, app_state

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "F3/F2 Racing Predictions API"
    assert data["status"] == "running"
    assert data["health"] == "/api/health"

    # Verify all expected keys are present
    expected_keys = {"name", "status", "health"}
    assert set(data.keys()) == expected_keys

    # Verify response content type
    assert response.headers["content-type"] == "application/json"


def test_health_check():
    # Mock app_state
    app_state.models = {"model1": None, "model2": None}
    app_state.system_status = {"last_training": datetime(2024, 1, 1)}

    with patch('main.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 15, 10, 30, 0)

        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["timestamp"] == "2024-01-15T10:30:00"
        assert data["models_loaded"] == 2
        assert data["last_training"] == "2024-01-01T00:00:00"

        # Verify response structure
        expected_keys = {"status", "timestamp", "models_loaded", "last_training"}
        assert set(data.keys()) == expected_keys
