"""
Unit tests for Policy Engine - v2 Backend
Tests role-based policy enforcement logic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.policy_engine import (
    check_policy,
    get_policy_options,
    format_policy_explanation,
    _is_alcohol_category
)


class TestPolicyEngine:
    """Test suite for policy engine."""
    
    def test_employee_alcohol_block(self):
        """Test that employees are blocked from alcohol purchases."""
        result = check_policy(
            user_role="employee",
            category="Alcohol & Beverages",
            amount=2500.0,
            merchant_type="verified",
            flags={}
        )
        
        assert result["action"] == "block"
        assert "EMPLOYEE_ALCOHOL_RESTRICT" in result["policy_reasons"]
    
    def test_employee_high_amount_warn(self):
        """Test that employees get warned on high amounts."""
        result = check_policy(
            user_role="employee",
            category="Business Travel",
            amount=55000.0,
            merchant_type="verified",
            flags={}
        )
        
        assert result["action"] == "warn"
        assert any("HIGH_AMOUNT_EMPLOYEE" in r for r in result["policy_reasons"])
    
    def test_employee_unverified_merchant_warn(self):
        """Test warning for unverified merchant with high amount."""
        result = check_policy(
            user_role="employee",
            category="Restaurant & Dining",
            amount=15000.0,
            merchant_type="unknown",
            flags={}
        )
        
        assert result["action"] == "warn"
        assert any("UNVERIFIED_MERCHANT" in r for r in result["policy_reasons"])
    
    def test_manager_alcohol_note_required(self):
        """Test that managers need note for alcohol."""
        result = check_policy(
            user_role="manager",
            category="Alcohol & Beverages",
            amount=2500.0,
            merchant_type="verified",
            flags={}
        )
        
        assert result["action"] == "allow_with_note"
        assert "ALCOHOL_MANAGER_NOTE_REQUIRED" in result["policy_reasons"]
    
    def test_manager_higher_threshold(self):
        """Test that managers have higher amount threshold."""
        result = check_policy(
            user_role="manager",
            category="Business Travel",
            amount=55000.0,  # Would warn employee, but not manager
            merchant_type="verified",
            flags={}
        )
        
        # Should allow (threshold is 100k for managers)
        assert result["action"] == "allow"
    
    def test_universal_block_threshold(self):
        """Test that extremely high amounts are blocked for all roles."""
        for role in ["employee", "manager", "admin"]:
            result = check_policy(
                user_role=role,
                category="Business Travel",
                amount=250000.0,  # Above 200k threshold
                merchant_type="verified",
                flags={}
            )
            
            assert result["action"] == "block"
            assert any("AMOUNT_EXCEEDS_LIMIT" in r for r in result["policy_reasons"])
    
    def test_subscription_warning(self):
        """Test warning for unappro ved subscriptions."""
        result = check_policy(
            user_role="employee",
            category="Software & Services",
            amount=999.0,
            merchant_type="verified",
            flags={"is_subscription": True, "subscription_approved": False}
        )
        
        assert result["action"] == "warn"
        assert "SUBSCRIPTION_DETECTED_NO_APPROVAL" in result["policy_reasons"]
    
    def test_low_quality_note_warning(self):
        """Test warning for low quality notes."""
        result = check_policy(
            user_role="employee",
            category="Restaurant & Dining",
            amount=1200.0,
            merchant_type="verified",
            flags={"low_quality_note": True}
        )
        
        assert result["action"] == "warn"
        assert "LOW_QUALITY_NOTE" in result["policy_reasons"]
    
    def test_admin_no_restrictions(self):
        """Test that admins have minimal restrictions."""
        result = check_policy(
            user_role="admin",
            category="Alcohol & Beverages",
            amount=55000.0,
            merchant_type="unknown",
            flags={}
        )
        
        # Should allow (only universal rules apply)
        assert result["action"] == "allow"
    
    def test_policy_options_block(self):
        """Test options for block action."""
        options = get_policy_options("block")
        assert "Cancel" in options
        assert "Request Override" in options
    
    def test_policy_options_warn(self):
        """Test options for warn action."""
        options = get_policy_options("warn")
        assert "Continue Anyway" in options
        assert "Edit Note" in options
        assert "Cancel" in options
    
    def test_policy_options_allow(self):
        """Test options for allow action."""
        options = get_policy_options("allow")
        assert "Continue" in options
    
    def test_is_alcohol_category(self):
        """Test alcohol category detection."""
        assert _is_alcohol_category("Alcohol & Beverages")
        assert _is_alcohol_category("Wine Shop")
        assert _is_alcohol_category("Bar & Pub")
        assert not _is_alcohol_category("Restaurant & Dining")
        assert not _is_alcohol_category("Groceries")
    
    def test_format_policy_explanation(self):
        """Test policy explanation formatting."""
        explanation = format_policy_explanation(
            "block",
            ["EMPLOYEE_ALCOHOL_RESTRICT"]
        )
        
        assert "alcohol" in explanation.lower()
        assert "policy" in explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
