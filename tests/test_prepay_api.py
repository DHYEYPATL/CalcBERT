"""
API Integration tests for v2 Backend
Tests prepay check API endpoints with FastAPI TestClient.
"""

import pytest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


class TestPrepayAPI:
    """Test suite for prepay API endpoints."""
    
    def test_prepay_check_valid_request(self):
        """Test prepay check with valid request."""
        payload = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting + dinner",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "decision" in data
        assert "options" in data
        assert "analysis" in data
        assert "check_id" in data
        
        # Check decision is valid
        assert data["decision"] in ["allow", "warn", "block", "allow_with_note"]
        
        # Check analysis structure
        analysis = data["analysis"]
        assert "final_category" in analysis
        assert "final_confidence" in analysis
        assert "risk_flags" in analysis
        assert "explanation" in analysis
        assert "subscription" in analysis
    
    def test_prepay_check_high_amount_warning(self):
        """Test prepay check with high amount triggers warning."""
        payload = {
            "merchant": "Hotel Taj",
            "amount": 55000.0,
            "note": "conference accommodation",
            "upi_id": "taj@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should warn on high amount
        assert data["decision"] in ["warn", "block"]
        assert data["analysis"]["risk_flags"]["high_amount"] is True
    
    def test_prepay_check_alcohol_employee_block(self):
        """Test that employee alcohol purchase is blocked."""
        payload = {
            "merchant": "Wine Shop",
            "amount": 2500.0,
            "note": "team celebration",
            "upi_id": "wine@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be blocked or warned (depending on category detection)
        assert data["decision"] in ["block", "warn"]
    
    def test_prepay_check_manager_override(self):
        """Test that managers have different thresholds."""
        payload = {
            "merchant": "Hotel Taj",
            "amount": 55000.0,
            "note": "conference accommodation",
            "upi_id": "taj@upi",
            "user_role": "manager",
            "user_id": "manager-456",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Manager should have more lenient treatment
        # (might still warn but not block)
        assert data["decision"] in ["allow", "warn", "allow_with_note"]
    
    def test_prepay_check_low_quality_note(self):
        """Test that low quality notes trigger warnings."""
        payload = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "stuff",  # Low quality note
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should flag low quality note
        assert data["analysis"]["risk_flags"]["low_quality_note"] is True
    
    def test_prepay_check_subscription_detection(self):
        """Test subscription detection."""
        payload = {
            "merchant": "Netflix",
            "amount": 199.0,
            "note": "monthly premium subscription",
            "upi_id": "netflix@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should detect subscription
        assert data["analysis"]["subscription"]["is_subscription"] is True
    
    def test_prepay_check_invalid_amount(self):
        """Test validation for invalid amount."""
        payload = {
            "merchant": "Swiggy",
            "amount": -100.0,  # Invalid negative amount
            "note": "client meeting",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_prepay_check_missing_fields(self):
        """Test validation for missing required fields."""
        payload = {
            "merchant": "Swiggy",
            "amount": 1200.0
            # Missing other required fields
        }
        
        response = client.post("/prepay/check", json=payload)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_prepay_history_endpoint(self):
        """Test prepay history endpoint."""
        # First create a check
        payload = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "test-user-history",
            "date": "2025-12-24"
        }
        
        client.post("/prepay/check", json=payload)
        
        # Get history
        response = client.get("/prepay/history?user_id=test-user-history&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "count" in data
        assert "history" in data
        assert data["status"] == "ok"
        assert len(data["history"]) > 0
    
    def test_prepay_stats_endpoint(self):
        """Test prepay stats endpoint."""
        # Create a few checks
        for i in range(3):
            payload = {
                "merchant": "Swiggy",
                "amount": 1200.0 + (i * 100),
                "note": f"meeting {i}",
                "upi_id": "swiggy@upi",
                "user_role": "employee",
                "user_id": "test-user-stats",
                "date": "2025-12-24"
            }
            client.post("/prepay/check", json=payload)
        
        # Get stats
        response = client.get("/prepay/stats?user_id=test-user-stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "total_checks" in data
        assert "decision_breakdown" in data
        assert "total_amount" in data
        assert data["status"] == "ok"
    
    def test_root_endpoint_includes_prepay(self):
        """Test that root endpoint lists prepay endpoints."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "endpoints" in data
        endpoints = data["endpoints"]
        
        assert "prepay_check" in endpoints
        assert "prepay_history" in endpoints
        assert "prepay_stats" in endpoints
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "version" in data


class TestDemoPayloads:
    """Test with demo payloads from specification."""
    
    def test_demo_payload_1_normal_expense(self):
        """Test demo payload 1: Normal business expense."""
        payload = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting + dinner",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be allowed or warned (normal expense)
        assert data["decision"] in ["allow", "warn"]
        assert isinstance(data["options"], list)
        assert len(data["options"]) > 0
    
    def test_demo_payload_2_high_amount(self):
        """Test demo payload 2: High amount warning."""
        payload = {
            "merchant": "Hotel Taj",
            "amount": 55000.0,
            "note": "conference accommodation",
            "upi_id": "taj@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should warn on high amount
        assert data["decision"] in ["warn", "block"]
        assert data["analysis"]["risk_flags"]["high_amount"] is True
    
    def test_demo_payload_3_alcohol_block(self):
        """Test demo payload 3: Alcohol block."""
        payload = {
            "merchant": "Wine Shop",
            "amount": 2500.0,
            "note": "team celebration",
            "upi_id": "wine@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be blocked or warned
        assert data["decision"] in ["block", "warn"]
    
    def test_demo_payload_4_manager_override(self):
        """Test demo payload 4: Manager override."""
        payload = {
            "merchant": "Hotel Taj",
            "amount": 55000.0,
            "note": "conference accommodation",
            "upi_id": "taj@upi",
            "user_role": "manager",
            "user_id": "manager-456",
            "date": "2025-12-24"
        }
        
        response = client.post("/prepay/check", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Manager should have more lenient treatment
        assert data["decision"] in ["allow", "warn", "allow_with_note"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
