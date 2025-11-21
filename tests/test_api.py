"""
API Endpoint Tests for CalcBERT Backend
Tests all endpoints with mocked ML models.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app import app
from backend.model_adapter import ModelAdapter
from backend.storage import init_db, clear_feedback, save_feedback

# Create test client
client = TestClient(app)


@pytest.fixture(scope="module")
def setup_test_db():
    """Setup test database before tests."""
    init_db()
    yield
    # Cleanup after tests
    try:
        clear_feedback()
    except:
        pass


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Metrics file may or may not exist, so just check structure


def test_model_status_endpoint():
    """Test model status endpoint."""
    response = client.get("/model-status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_predict_endpoint_mock(monkeypatch):
    """Test predict endpoint with mocked model."""
    
    def mock_predict(self, text, meta=None):
        """Mock prediction function."""
        return {
            "label": "Coffee & Beverages",
            "confidence": 0.94,
            "rationale": {
                "rule_hits": ["starbucks"],
                "top_tokens": [
                    {"token": "starbucks", "score": 0.5},
                    {"token": "coffee", "score": 0.3}
                ]
            },
            "model_used": "tfidf"
        }
    
    # Monkeypatch the predict method
    monkeypatch.setattr(ModelAdapter, "predict", mock_predict)
    
    # Test prediction
    response = client.post(
        "/predict",
        json={"text": "STARBCKS #1023 MUMBAI"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Coffee & Beverages"
    assert data["confidence"] == 0.94
    assert "explanation" in data
    assert "model_used" in data


def test_predict_endpoint_with_meta(monkeypatch):
    """Test predict endpoint with metadata."""
    
    def mock_predict(self, text, meta=None):
        return {
            "label": "Transportation",
            "confidence": 0.88,
            "rationale": {
                "rule_hits": ["uber"],
                "top_tokens": [{"token": "uber", "score": 0.6}]
            },
            "model_used": "fusion"
        }
    
    monkeypatch.setattr(ModelAdapter, "predict", mock_predict)
    
    response = client.post(
        "/predict",
        json={
            "text": "UBER TRIP 12:30PM",
            "meta": {"mcc": "4121", "time": "12:30PM"}
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Transportation"


def test_predict_endpoint_invalid_input():
    """Test predict endpoint with invalid input."""
    response = client.post(
        "/predict",
        json={"text": ""}  # Empty text
    )
    assert response.status_code == 422  # Validation error


def test_feedback_endpoint(setup_test_db):
    """Test feedback storage endpoint."""
    response = client.post(
        "/feedback",
        json={
            "text": "STARBCKS #1023",
            "correct_label": "Coffee & Beverages",
            "user_id": "test_user"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"
    assert "id" in data
    assert isinstance(data["id"], int)


def test_feedback_count_endpoint(setup_test_db):
    """Test feedback count endpoint."""
    # Add some feedback
    save_feedback("Test transaction", "Test Category", "test_user")
    
    response = client.get("/feedback/count")
    assert response.status_code == 200
    data = response.json()
    assert "total_feedback" in data
    assert data["total_feedback"] >= 1


def test_feedback_endpoint_invalid_input():
    """Test feedback endpoint with invalid input."""
    response = client.post(
        "/feedback",
        json={
            "text": "",  # Empty text
            "correct_label": "Coffee"
        }
    )
    assert response.status_code == 422  # Validation error


def test_retrain_status_endpoint():
    """Test retrain status endpoint."""
    response = client.get("/retrain/status")
    assert response.status_code == 200
    data = response.json()
    assert "sync_mode" in data
    assert "supported_models" in data


def test_retrain_endpoint_invalid_model():
    """Test retrain endpoint with unsupported model."""
    response = client.post(
        "/retrain",
        json={
            "mode": "incremental",
            "model": "distilbert"  # Not supported via API
        }
    )
    assert response.status_code == 400


def test_retrain_endpoint_invalid_mode():
    """Test retrain endpoint with unsupported mode."""
    response = client.post(
        "/retrain",
        json={
            "mode": "full",  # Not supported
            "model": "tfidf"
        }
    )
    assert response.status_code == 400


def test_full_workflow_mock(monkeypatch, setup_test_db):
    """Test complete workflow: predict -> feedback -> retrain."""
    
    # Mock prediction
    def mock_predict(self, text, meta=None):
        return {
            "label": "Shopping",
            "confidence": 0.85,
            "rationale": {
                "rule_hits": [],
                "top_tokens": [{"token": "amazon", "score": 0.4}]
            },
            "model_used": "tfidf"
        }
    
    monkeypatch.setattr(ModelAdapter, "predict", mock_predict)
    
    # 1. Predict
    pred_response = client.post(
        "/predict",
        json={"text": "AMAZON.COM PURCHASE"}
    )
    assert pred_response.status_code == 200
    
    # 2. Submit feedback
    feedback_response = client.post(
        "/feedback",
        json={
            "text": "AMAZON.COM PURCHASE",
            "correct_label": "Online Shopping",
            "user_id": "test_user"
        }
    )
    assert feedback_response.status_code == 200
    
    # 3. Check feedback count
    count_response = client.get("/feedback/count")
    assert count_response.status_code == 200
    assert count_response.json()["total_feedback"] >= 1
    
    # Note: Retrain test would require ML modules to be present
    # For now, just test the endpoint structure
    status_response = client.get("/retrain/status")
    assert status_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
