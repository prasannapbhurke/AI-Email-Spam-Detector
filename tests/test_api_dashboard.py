"""Tests for dashboard and stats endpoints."""

from fastapi.testclient import TestClient
from spam_detector.api.main import app


def test_dashboard_stats(client: TestClient):
    """Test dashboard stats endpoint."""
    response = client.get("/dashboard/stats")
    assert response.status_code == 200

    data = response.json()
    assert "total_predictions" in data
    assert "spam_predictions" in data
    assert "ham_predictions" in data
    assert "spam_rate" in data


def test_dashboard_accuracy_over_time(client: TestClient):
    """Test accuracy over time endpoint."""
    response = client.get("/dashboard/accuracy-over-time?days=7")
    assert response.status_code == 200

    data = response.json()
    assert "bucket" in data
    assert "period_days" in data
    assert "data" in data


def test_dashboard_model_performance(client: TestClient):
    """Test model performance endpoint."""
    response = client.get("/dashboard/model-performance")
    assert response.status_code == 200

    data = response.json()
    assert "metrics" in data
    if "metrics" in data:
        metrics = data["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


def test_dashboard_spam_stats(client: TestClient):
    """Test spam statistics endpoint."""
    response = client.get("/dashboard/spam-stats")
    assert response.status_code == 200


def test_feedback_analytics(client: TestClient):
    """Test feedback analytics endpoint."""
    response = client.get("/dashboard/feedback-analytics")
    assert response.status_code == 200
