"""Health endpoint tests (scaffold)."""

from fastapi.testclient import TestClient

from clauseguard.api.main import create_app


def test_health_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "qdrant" in body
