"""Tests for prediction API endpoint."""

from fastapi.testclient import TestClient
from spam_detector.api.main import app


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"  # model and DB should be ready in test client
    assert data["model_ready"] is True
    assert data["db_ready"] is True


def test_predict_endpoint_valid(client: TestClient):
    """Test prediction with valid email text."""
    payload = {
        "email_text": "Test email content for spam detection"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert "prediction" in data
    assert data["prediction"] in ["spam", "ham"]
    assert "confidence" in data
    assert "spam_probability" in data
    assert "predicted_at" in data


def test_predict_endpoint_empty(client: TestClient):
    """Test prediction with empty email text."""
    payload = {
        "email_text": ""
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # validation error


def test_predict_endpoint_missing_field(client: TestClient):
    """Test prediction with missing email_text field."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_spam_high_confidence(client: TestClient):
    """Test that obvious spam is detected with high confidence."""
    payload = {
        "email_text": "WIN FREE MONEY NOW!!! CLICK HERE TO CLAIM YOUR PRIZE!!!"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["confidence"] > 0.7


def test_predict_ham_low_confidence(client: TestClient):
    """Test that legitimate email is classified as not spam."""
    payload = {
        "email_text": "Hi John, can we meet tomorrow at 2pm to discuss the project?"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "not_spam"
    assert data["confidence"] > 0.5
