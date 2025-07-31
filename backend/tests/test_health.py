from fastapi.testclient import TestClient
import app
from app.main import create_app

client = TestClient(app)


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
