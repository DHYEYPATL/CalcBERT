"""
Prepay Check API Route - v2 Backend
Handles prepay expense approval checks with policy enforcement.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.langgraph_workflow import run_workflow
from backend.storage import save_prepay_check, get_prepay_checks, get_prepay_check_count

router = APIRouter()


class PrepayCheckRequest(BaseModel):
    """Request schema for prepay check endpoint."""
    merchant: str = Field(..., description="Merchant name", min_length=1)
    amount: float = Field(..., description="Transaction amount", gt=0)
    note: str = Field(..., description="Transaction note/description")
    upi_id: str = Field(..., description="UPI ID for payment")
    user_role: str = Field(..., description="User role (employee, manager, admin)")
    user_id: str = Field(..., description="User identifier")
    date: str = Field(..., description="Transaction date (ISO format)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "merchant": "Swiggy",
                "amount": 1200.0,
                "note": "client meeting + dinner",
                "upi_id": "swiggy@upi",
                "user_role": "employee",
                "user_id": "user-123",
                "date": "2025-12-24"
            }
        }


class AnalysisDetails(BaseModel):
    """Analysis details in response."""
    final_category: str
    final_confidence: float
    risk_flags: Dict[str, bool]
    explanation: Dict[str, Any]
    subscription: Dict[str, Any]
    similar_expenses: List[Dict[str, Any]] = []


class PrepayCheckResponse(BaseModel):
    """Response schema for prepay check endpoint."""
    decision: str = Field(..., description="Policy decision (allow, warn, block, allow_with_note)")
    options: List[str] = Field(..., description="Available user options")
    analysis: AnalysisDetails = Field(..., description="Detailed analysis")
    check_id: Optional[int] = Field(None, description="Database ID of this check")


@router.post("/check", response_model=PrepayCheckResponse)
def prepay_check(req: PrepayCheckRequest) -> PrepayCheckResponse:
    """
    Check prepay expense against policies.
    
    Args:
        req: PrepayCheckRequest with expense details
        
    Returns:
        PrepayCheckResponse with decision and analysis
        
    Raises:
        HTTPException: If check fails
    """
    try:
        # Build initial state
        state = {
            "merchant": req.merchant,
            "amount": req.amount,
            "note": req.note,
            "upi_id": req.upi_id,
            "user_role": req.user_role,
            "user_id": req.user_id,
            "date": req.date,
            "meta": {}
        }
        
        # Run workflow
        final_state = run_workflow(state)
        
        # Extract results
        fusion_result = final_state.get("fusion_result", {})
        policy_result = final_state.get("policy_result", {})
        merchant_status = final_state.get("merchant_status", {})
        explanation = final_state.get("explanation", {})
        
        # Debug: Log fusion_result to help diagnose issues
        if not fusion_result or not fusion_result.get("label") and not fusion_result.get("final_category"):
            print(f"⚠ Warning: fusion_result is empty or missing category keys: {fusion_result}")
            print(f"   Available keys in final_state: {list(final_state.keys())}")
        
        # Extract category and confidence from fusion_result
        # Handle both "label"/"confidence" (from workflow) and "final_category"/"final_confidence" (legacy)
        category = (
            fusion_result.get("final_category") or 
            fusion_result.get("label") or 
            "Unknown"
        )
        # Normalize "unknown" (lowercase) to "Unknown" for consistency
        if category and category.lower() == "unknown":
            category = "Unknown"
            
        confidence = float(
            fusion_result.get("final_confidence") or 
            fusion_result.get("confidence") or 
            0.0
        )
        
        # Log extracted values for debugging
        print(f"✓ Extracted category: '{category}', confidence: {confidence}")
        print(f"   Full fusion_result: {fusion_result}")
        
        # Build risk flags
        risk_flags = {
            "high_amount": req.amount > 50000.0,
            "unverified_merchant": merchant_status.get("merchant_type") == "unknown",
            "low_quality_note": merchant_status.get("note_quality", {}).get("is_low_quality", False),
            "policy_violation": policy_result.get("action") in ["block", "warn"]
        }
        
        # Build analysis
        analysis = AnalysisDetails(
            final_category=category,
            final_confidence=confidence,
            risk_flags=risk_flags,
            explanation=explanation,
            subscription=merchant_status.get("subscription", {
                "is_subscription": False,
                "confidence": 0.0
            }),
            similar_expenses=[]  # TODO: Implement vector store query
        )
        
        # Get decision and options
        decision = final_state.get("final_decision", "allow")
        options = policy_result.get("options", ["Continue"])
        
        # Persist to database
        analysis_json = json.dumps({
            "final_category": analysis.final_category,
            "final_confidence": analysis.final_confidence,
            "risk_flags": analysis.risk_flags,
            "explanation": analysis.explanation,
            "subscription": analysis.subscription,
            "merchant_status": merchant_status,
            "fusion_result": fusion_result,
            "policy_result": policy_result
        })
        
        check_id = save_prepay_check(
            user_id=req.user_id,
            merchant=req.merchant,
            amount=req.amount,
            note=req.note,
            upi_id=req.upi_id,
            user_role=req.user_role,
            date=req.date,
            decision=decision,
            analysis_json=analysis_json
        )
        
        # Return response
        return PrepayCheckResponse(
            decision=decision,
            options=options,
            analysis=analysis,
            check_id=check_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prepay check failed: {str(e)}"
        )


@router.get("/history")
def get_prepay_history(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),  # Backward compatibility
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get prepay check history.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        limit: Maximum number of records to return
        
    Returns:
        Dictionary with history data
    """
    try:
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        checks = get_prepay_checks(user_id=effective_user_id, limit=limit)
        
        # Format results
        history = []
        for check in checks:
            history.append({
                "id": check[0],
                "user_id": check[1],
                "merchant": check[2],
                "amount": check[3],
                "note": check[4],
                "upi_id": check[5],
                "user_role": check[6],
                "date": check[7],
                "decision": check[8],
                "created_at": check[10]
            })
        
        return {
            "status": "ok",
            "count": len(history),
            "total": get_prepay_check_count(user_id=effective_user_id),
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/stats")
def get_prepay_stats(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get prepay check statistics.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        Dictionary with statistics
    """
    try:
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        total = get_prepay_check_count(user_id=effective_user_id)
        
        # Get recent checks for stats
        checks = get_prepay_checks(user_id=effective_user_id, limit=100)
        
        # Count by decision
        decision_counts = {"allow": 0, "warn": 0, "block": 0, "allow_with_note": 0}
        total_amount = 0.0
        
        for check in checks:
            decision = check[8]
            amount = check[3]
            
            if decision in decision_counts:
                decision_counts[decision] += 1
            total_amount += amount
        
        return {
            "status": "ok",
            "total_checks": total,
            "recent_checks": len(checks),
            "decision_breakdown": decision_counts,
            "total_amount": total_amount,
            "average_amount": total_amount / len(checks) if checks else 0.0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve stats: {str(e)}"
        )
