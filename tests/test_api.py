from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_version() -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["version"]
    assert response.json()["environment"]


def test_epias_health_does_not_expose_credentials() -> None:
    response = client.get("/api/epias/health")

    assert response.status_code == 200
    assert response.json()["client_ready"] is True
    assert "epias_base_url" in response.json()
    assert "username" not in response.json()
    assert "password" not in response.json()
