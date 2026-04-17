"""Tests for prediction API endpoint."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from fastapi.testclient import TestClient
from spam_detector.api.main import app


client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint_valid():
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


def test_predict_endpoint_empty():
    """Test prediction with empty email text."""
    payload = {
        "email_text": ""
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # validation error


def test_predict_endpoint_missing_field():
    """Test prediction with missing email_text field."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_spam_high_confidence():
    """Test that obvious spam is detected with high confidence."""
    payload = {
        "email_text": "WIN FREE MONEY NOW!!! CLICK HERE TO CLAIM YOUR PRIZE!!!"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["confidence"] > 0.7


def test_predict_ham_low_confidence():
    """Test that legitimate email is classified as not spam."""
    payload = {
        "email_text": "Hi John, can we meet tomorrow at 2pm to discuss the project?"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "ham"
    assert data["confidence"] > 0.5
