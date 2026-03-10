"""API integration tests using FastAPI TestClient."""

from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_simulate_orbital_api():
    payload = {"name": "2p0", "n_points": 1000, "seed": 123}
    response = client.post("/simulate/orbital", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_points"] == 1000
    assert "points" in data
    assert "values" in data

def test_invalid_orbital_error():
    payload = {"name": "invalid_orbital", "n_points": 1000}
    response = client.post("/simulate/orbital", json=payload)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
