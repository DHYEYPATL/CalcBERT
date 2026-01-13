"""
Policy Engine for v2 Backend - Role-based expense approval policies.
Enforces business rules based on user role, category, amount, and merchant verification.
"""

from typing import Dict, Any, List
from backend.config import settings


def check_policy(
    user_role: str,
    category: str,
    amount: float,
    merchant_type: str,
    flags: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check policy rules and return action decision.
    
    Args:
        user_role: User role ('employee', 'manager', 'admin')
        category: Predicted expense category
        amount: Transaction amount
        merchant_type: Merchant verification status ('verified', 'known', 'unknown')
        flags: Additional risk flags from analysis
        
    Returns:
        Dictionary with:
        {
            "action": "allow"|"warn"|"block"|"allow_with_note",
            "policy_reasons": list[str]
        }
    """
    policy_reasons = []
    action = "allow"
    
    # Universal rules (apply to all roles)
    universal_action, universal_reasons = _check_universal_policies(amount, category, flags)
    if universal_action != "allow":
        action = universal_action
        policy_reasons.extend(universal_reasons)
    
    # Role-specific rules
    if user_role.lower() == "employee":
        role_action, role_reasons = _check_employee_policies(category, amount, merchant_type, flags)
        if role_action != "allow":
            # Escalate action if more restrictive
            if _is_more_restrictive(role_action, action):
                action = role_action
            policy_reasons.extend(role_reasons)
    
    elif user_role.lower() == "manager":
        role_action, role_reasons = _check_manager_policies(category, amount, merchant_type, flags)
        if role_action != "allow":
            if _is_more_restrictive(role_action, action):
                action = role_action
            policy_reasons.extend(role_reasons)
    
    # Admin has no additional restrictions beyond universal
    
    return {
        "action": action,
        "policy_reasons": policy_reasons
    }


def _check_universal_policies(
    amount: float,
    category: str,
    flags: Dict[str, Any]
) -> tuple[str, List[str]]:
    """
    Check universal policies that apply to all users.
    
    Returns:
        (action, reasons)
    """
    reasons = []
    action = "allow"
    
    # Block extremely high amounts (requires special approval)
    if amount > settings.UNIVERSAL_BLOCK_THRESHOLD:
        action = "block"
        reasons.append(f"AMOUNT_EXCEEDS_LIMIT (₹{settings.UNIVERSAL_BLOCK_THRESHOLD:,.0f})")
    
    # Warn on subscription without approval flag
    if flags.get("is_subscription", False) and not flags.get("subscription_approved", False):
        if action == "allow":
            action = "warn"
        reasons.append("SUBSCRIPTION_DETECTED_NO_APPROVAL")
    
    return action, reasons


def _check_employee_policies(
    category: str,
    amount: float,
    merchant_type: str,
    flags: Dict[str, Any]
) -> tuple[str, List[str]]:
    """
    Check employee-specific policies.
    
    Returns:
        (action, reasons)
    """
    reasons = []
    action = "allow"
    
    # Block alcohol purchases for employees
    if settings.EMPLOYEE_ALCOHOL_BLOCK and _is_alcohol_category(category):
        action = "block"
        reasons.append("EMPLOYEE_ALCOHOL_RESTRICT")
    
    # Warn on high amounts
    if amount > settings.EMPLOYEE_HIGH_AMOUNT_THRESHOLD:
        if action == "allow":
            action = "warn"
        reasons.append(f"HIGH_AMOUNT_EMPLOYEE (>₹{settings.EMPLOYEE_HIGH_AMOUNT_THRESHOLD:,.0f})")
    
    # Warn on unverified merchant with significant amount
    if merchant_type == "unknown" and amount > settings.UNVERIFIED_MERCHANT_WARN_THRESHOLD:
        if action == "allow":
            action = "warn"
        reasons.append(f"UNVERIFIED_MERCHANT_HIGH_AMOUNT (>₹{settings.UNVERIFIED_MERCHANT_WARN_THRESHOLD:,.0f})")
    
    # Warn on low quality note
    if flags.get("low_quality_note", False):
        if action == "allow":
            action = "warn"
        reasons.append("LOW_QUALITY_NOTE")
    
    return action, reasons


def _check_manager_policies(
    category: str,
    amount: float,
    merchant_type: str,
    flags: Dict[str, Any]
) -> tuple[str, List[str]]:
    """
    Check manager-specific policies.
    Managers have more lenient rules but still some restrictions.
    
    Returns:
        (action, reasons)
    """
    reasons = []
    action = "allow"
    
    # Managers can approve alcohol but should note it
    if _is_alcohol_category(category):
        action = "allow_with_note"
        reasons.append("ALCOHOL_MANAGER_NOTE_REQUIRED")
    
    # Higher threshold for warnings
    manager_high_threshold = settings.EMPLOYEE_HIGH_AMOUNT_THRESHOLD * 2
    if amount > manager_high_threshold:
        if action == "allow":
            action = "warn"
        reasons.append(f"HIGH_AMOUNT_MANAGER (>₹{manager_high_threshold:,.0f})")
    
    # Note required for unverified merchants with high amounts
    if merchant_type == "unknown" and amount > settings.UNVERIFIED_MERCHANT_WARN_THRESHOLD * 2:
        if action == "allow":
            action = "allow_with_note"
        reasons.append("UNVERIFIED_MERCHANT_NOTE_REQUIRED")
    
    return action, reasons


def _is_alcohol_category(category: str) -> bool:
    """
    Check if category is alcohol-related.
    
    Args:
        category: Expense category
        
    Returns:
        True if alcohol-related
    """
    alcohol_keywords = ["alcohol", "wine", "beer", "liquor", "bar", "pub", "brewery"]
    category_lower = category.lower()
    return any(keyword in category_lower for keyword in alcohol_keywords)


def _is_more_restrictive(action1: str, action2: str) -> bool:
    """
    Compare two actions and return True if action1 is more restrictive than action2.
    
    Restriction order: block > warn > allow_with_note > allow
    
    Args:
        action1: First action
        action2: Second action
        
    Returns:
        True if action1 is more restrictive
    """
    restriction_order = {
        "allow": 0,
        "allow_with_note": 1,
        "warn": 2,
        "block": 3
    }
    
    return restriction_order.get(action1, 0) > restriction_order.get(action2, 0)


def get_policy_options(action: str) -> List[str]:
    """
    Get user options based on policy action.
    
    Args:
        action: Policy action
        
    Returns:
        List of available options
    """
    if action == "block":
        return ["Cancel", "Request Override"]
    elif action == "warn":
        return ["Continue Anyway", "Edit Note", "Cancel"]
    elif action == "allow_with_note":
        return ["Add Note and Continue", "Cancel"]
    else:  # allow
        return ["Continue"]


def format_policy_explanation(action: str, reasons: List[str]) -> str:
    """
    Format policy reasons into human-readable explanation.
    
    Args:
        action: Policy action
        reasons: List of policy reason codes
        
    Returns:
        Human-readable explanation
    """
    if not reasons:
        return "No policy restrictions apply."
    
    explanations = {
        "EMPLOYEE_ALCOHOL_RESTRICT": "Company policy prohibits alcohol purchases by employees.",
        "ALCOHOL_MANAGER_NOTE_REQUIRED": "Alcohol purchases require a business justification note.",
        "HIGH_AMOUNT_EMPLOYEE": "This amount requires manager approval.",
        "HIGH_AMOUNT_MANAGER": "This amount is unusually high and may require additional review.",
        "AMOUNT_EXCEEDS_LIMIT": "This amount exceeds the maximum allowed limit and requires special approval.",
        "UNVERIFIED_MERCHANT_HIGH_AMOUNT": "High-value transaction with unverified merchant requires review.",
        "UNVERIFIED_MERCHANT_NOTE_REQUIRED": "Unverified merchant requires business justification.",
        "LOW_QUALITY_NOTE": "Transaction note lacks sufficient detail for approval.",
        "SUBSCRIPTION_DETECTED_NO_APPROVAL": "Recurring subscription detected without prior approval."
    }
    
    formatted_reasons = []
    for reason in reasons:
        # Extract base reason without amount details
        base_reason = reason.split(" (")[0]
        explanation = explanations.get(base_reason, reason)
        formatted_reasons.append(f"• {explanation}")
    
    return "\n".join(formatted_reasons)
