import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_single_endpoint():
    payload = {
        "age": 35,
        "sex": "female",
        "bmi": 28.5,
        "children": 1,
        "smoker": "no",
        "region": "southwest"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_cost_usd" in data
    assert data["predicted_cost_usd"] > 0
    assert "confidence_interval_usd" in data
    assert "low_10th_percentile" in data["confidence_interval_usd"]
    assert "high_90th_percentile" in data["confidence_interval_usd"]

def test_history_endpoint():
    response = client.get("/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
