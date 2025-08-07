from fastapi.testclient import TestClient
from app.main import create_app
from app.dependencies import get_app_state


def test_root_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/")

    assert response.status_code == 200
    assert response.json() == {
        "name": "Formula Predictions API",
        "status": "running",
        "health": "/api/health"
    }


def test_health_check():
    app = create_app()
    client = TestClient(app)

    class MockAppState:
        models = ["model1", "model2"]
        system_status = {"last_training": "2023-01-01T12:00:00"}

    app.dependency_overrides[get_app_state] = lambda: MockAppState()

    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["models_loaded"] == 2
    assert data["last_training"] == "2023-01-01T12:00:00"
    assert "timestamp" in data
