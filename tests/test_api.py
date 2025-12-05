"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "uptime_seconds" in data


def test_get_available_stocks():
    """Test getting available stocks."""
    response = client.get("/api/v1/stocks/available")
    assert response.status_code == 200
    data = response.json()
    assert "stocks" in data
    assert "count" in data


def test_models_status():
    """Test models status endpoint."""
    response = client.get("/api/v1/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "count" in data


def test_prediction_without_model():
    """Test prediction endpoint without trained model."""
    response = client.post(
        "/api/v1/predict",
        json={"symbol": "NONEXISTENT", "days_ahead": 1}
    )
    # Should return 404 if model doesn't exist
    assert response.status_code in [404, 500]


def test_invalid_prediction_request():
    """Test prediction with invalid request."""
    response = client.post(
        "/api/v1/predict",
        json={"symbol": "AAPL", "days_ahead": 100}  # Too many days
    )
    # Should return validation error
    assert response.status_code == 422

