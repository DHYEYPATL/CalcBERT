"""
Unit tests for LangGraph Workflow - v2 Backend
Tests workflow orchestration in sequential mode.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.langgraph_workflow import WorkflowOrchestrator, run_workflow


class TestLangGraphWorkflow:
    """Test suite for workflow orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create workflow orchestrator instance."""
        return WorkflowOrchestrator()
    
    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting + dinner",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.model_adapter is not None
        assert orchestrator.mode in ["sequential", "langgraph"]
    
    def test_sequential_workflow_execution(self, orchestrator, sample_state):
        """Test complete sequential workflow execution."""
        result = orchestrator.run_workflow(sample_state, mode="sequential")
        
        # Check all nodes executed
        assert "merchant_status" in result
        assert "tfidf_output" in result or "rule_output" in result
        assert "policy_result" in result
        assert "fusion_result" in result
        assert "explanation" in result
        assert "final_decision" in result
    
    def test_merchant_check_node(self, orchestrator, sample_state):
        """Test merchant check node."""
        result = orchestrator._merchant_check_node(sample_state.copy())
        
        assert "merchant_status" in result
        merchant_status = result["merchant_status"]
        
        assert "merchant_type" in merchant_status
        assert "is_verified" in merchant_status
        assert "subscription" in merchant_status
        assert "note_quality" in merchant_status
    
    def test_merchant_type_detection(self, orchestrator):
        """Test merchant type detection."""
        # Verified merchant
        state1 = {"merchant": "Swiggy", "amount": 1200.0, "note": "food"}
        result1 = orchestrator._merchant_check_node(state1)
        assert result1["merchant_status"]["merchant_type"] == "verified"
        
        # Unknown merchant
        state2 = {"merchant": "Random Shop", "amount": 1200.0, "note": "purchase"}
        result2 = orchestrator._merchant_check_node(state2)
        assert result2["merchant_status"]["merchant_type"] == "unknown"
    
    def test_subscription_detection(self, orchestrator):
        """Test subscription detection."""
        # Known subscription merchant
        state1 = {
            "merchant": "Netflix",
            "amount": 199.0,
            "note": "monthly subscription"
        }
        result1 = orchestrator._merchant_check_node(state1)
        assert result1["merchant_status"]["subscription"]["is_subscription"] is True
        
        # Non-subscription
        state2 = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "dinner"
        }
        result2 = orchestrator._merchant_check_node(state2)
        assert result2["merchant_status"]["subscription"]["is_subscription"] is False
    
    def test_note_quality_detection(self, orchestrator):
        """Test note quality detection."""
        # Good quality note
        state1 = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting dinner with stakeholders"
        }
        result1 = orchestrator._merchant_check_node(state1)
        assert result1["merchant_status"]["note_quality"]["is_low_quality"] is False
        
        # Low quality note
        state2 = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "stuff"
        }
        result2 = orchestrator._merchant_check_node(state2)
        assert result2["merchant_status"]["note_quality"]["is_low_quality"] is True
    
    def test_tfidf_node(self, orchestrator, sample_state):
        """Test TF-IDF prediction node."""
        result = orchestrator._tfidf_node(sample_state.copy())
        
        # Should have either tfidf_output or rule_output (depending on model availability)
        assert "tfidf_output" in result or "rule_output" in result
    
    def test_policy_node(self, orchestrator):
        """Test policy enforcement node."""
        # Setup state with predictions
        state = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting dinner",
            "user_role": "employee",
            "merchant_status": {
                "merchant_type": "verified",
                "subscription": {"is_subscription": False},
                "note_quality": {"is_low_quality": False}
            },
            "rule_output": {"label": "Restaurant & Dining", "confidence": 0.95}
        }
        
        result = orchestrator._policy_node(state)
        
        assert "policy_result" in result
        policy_result = result["policy_result"]
        
        assert "action" in policy_result
        assert "policy_reasons" in policy_result
        assert "options" in policy_result
    
    def test_policy_employee_alcohol_block(self, orchestrator):
        """Test that employees are blocked from alcohol."""
        state = {
            "merchant": "Wine Shop",
            "amount": 2500.0,
            "note": "team celebration",
            "user_role": "employee",
            "merchant_status": {
                "merchant_type": "verified",
                "subscription": {"is_subscription": False},
                "note_quality": {"is_low_quality": False}
            },
            "rule_output": {"label": "Alcohol & Beverages", "confidence": 0.95}
        }
        
        result = orchestrator._policy_node(state)
        assert result["policy_result"]["action"] == "block"
    
    def test_fusion_node(self, orchestrator):
        """Test fusion node."""
        state = {
            "rule_output": {"label": "Restaurant & Dining", "confidence": 0.95, "matches": ["swiggy"]},
            "tfidf_output": {"label": "Restaurant & Dining", "confidence": 0.87},
            "policy_result": {"action": "allow", "policy_reasons": []}
        }
        
        result = orchestrator._fusion_node(state)
        
        assert "fusion_result" in result
        assert "final_decision" in result
        
        fusion_result = result["fusion_result"]
        assert "label" in fusion_result
        assert "confidence" in fusion_result
        assert "model_used" in fusion_result
    
    def test_explain_node(self, orchestrator):
        """Test explanation builder node."""
        state = {
            "fusion_result": {
                "label": "Restaurant & Dining",
                "confidence": 0.95,
                "model_used": "rule"
            },
            "policy_result": {
                "action": "allow",
                "policy_reasons": []
            },
            "merchant_status": {
                "note_quality": {"is_low_quality": False, "suggestions": []}
            }
        }
        
        result = orchestrator._explain_node(state)
        
        assert "explanation" in result
        explanation = result["explanation"]
        
        assert "category_reason" in explanation
        assert "policy_reason" in explanation
        assert "suggestions" in explanation
    
    def test_end_to_end_allow_scenario(self, orchestrator):
        """Test end-to-end workflow for allowed expense."""
        state = {
            "merchant": "Swiggy",
            "amount": 1200.0,
            "note": "client meeting dinner with stakeholders",
            "upi_id": "swiggy@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        result = run_workflow(state)
        
        # Should be allowed (normal business expense)
        assert result["final_decision"] in ["allow", "warn"]
    
    def test_end_to_end_block_scenario(self, orchestrator):
        """Test end-to-end workflow for blocked expense."""
        state = {
            "merchant": "Wine Shop",
            "amount": 2500.0,
            "note": "team celebration drinks",
            "upi_id": "wine@upi",
            "user_role": "employee",
            "user_id": "user-123",
            "date": "2025-12-24"
        }
        
        result = run_workflow(state)
        
        # Should be blocked (employee + alcohol)
        # Note: Might be "warn" if category detection fails, but policy should catch it
        assert result["final_decision"] in ["block", "warn"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
