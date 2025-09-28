from fastapi.testclient import TestClient
from app.main import create_app
from app.dependencies import get_app_state


def test_root_endpoint():
    app = create_app()
    client = TestClient(app)

    root_path = str(app.url_path_for("root"))
    health_path = str(app.url_path_for("health_check"))

    response = client.get(root_path)
    assert response.status_code == 200
    assert response.json() == {
        "name": "Formula Predictions API",
        "status": "running",
        "health": health_path
    }


def test_health_check():
    app = create_app()
    client = TestClient(app)

    class MockAppState:
        models = ["model1", "model2"]
        system_status = {"last_training": "2023-01-01T12:00:00"}

    app.dependency_overrides[get_app_state] = lambda: MockAppState()

    health_path = app.url_path_for("health_check")
    response = client.get(health_path)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["models_loaded"] == 2
    assert data["last_training"] == "2023-01-01T12:00:00"
    assert "timestamp" in data


def test_health_head():
    app = create_app()
    client = TestClient(app)

    head_path = str(app.url_path_for("health_head"))
    response = client.request("HEAD", head_path)

    assert response.status_code == 200
    assert response.content == b""
