"""
Summary and Subscription API Routes - v2 Backend
Provides daily summaries, subscription management, and alerts.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd

from backend.storage import get_prepay_checks

router = APIRouter()


@router.get("/today")
def get_daily_summary(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get daily spend summary.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        Daily summary with spend by category, confidence, etc.
    """
    try:
        from ml.summary_builder import build_daily_summary
        
        # Get today's transactions
        checks = get_prepay_checks(user_id=user_id, limit=1000)
        
        # Filter for today
        today = datetime.now().date()
        today_checks = []
        
        for check in checks:
            # check[10] is created_at timestamp
            check_date = datetime.fromtimestamp(check[10]).date()
            if check_date == today:
                today_checks.append(check)
        
        # Convert to DataFrame for summary_builder
        if today_checks:
            df = pd.DataFrame(today_checks, columns=[
                "id", "user_id", "merchant", "amount", "note",
                "upi_id", "user_role", "date", "decision", "analysis_json", "created_at"
            ])
            
            # Add category and confidence from analysis_json
            import json
            df["category"] = df["analysis_json"].apply(
                lambda x: json.loads(x).get("final_category", "Unknown")
            )
            df["confidence"] = df["analysis_json"].apply(
                lambda x: json.loads(x).get("final_confidence", 0.0)
            )
            df["corrected"] = False  # No correction tracking yet
            
            summary = build_daily_summary(df)
        else:
            summary = {
                "today_spend_by_category": {},
                "average_confidence": None,
                "manual_corrections": 0,
                "top_merchants": []
            }
        
        summary["total_transactions"] = len(today_checks)
        summary["date"] = str(today)
        
        return {
            "status": "ok",
            "summary": summary
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build summary: {str(e)}"
        )


@router.get("/subscriptions")
def get_subscriptions(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get detected subscriptions.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        List of detected subscriptions
    """
    try:
        from ml.subscription_detector import detect_subscription
        
        # Get all transactions
        checks = get_prepay_checks(user_id=user_id, limit=1000)
        
        # Convert to DataFrame
        if checks:
            df = pd.DataFrame(checks, columns=[
                "id", "user_id", "merchant", "amount", "note",
                "upi_id", "user_role", "date", "decision", "analysis_json", "created_at"
            ])
            
            # Group by merchant and check for subscriptions
            subscriptions = []
            
            for merchant in df["merchant"].unique():
                merchant_df = df[df["merchant"] == merchant]
                
                # Use most common amount for this merchant
                common_amount = merchant_df["amount"].mode()[0] if len(merchant_df) > 0 else 0
                
                # Detect subscription
                result = detect_subscription(df, merchant, common_amount)
                
                if result["is_subscription"]:
                    subscriptions.append({
                        "merchant": merchant,
                        "amount": float(common_amount),
                        "period": result.get("period"),
                        "count": result.get("count", 0),
                        "reason": result.get("reason", "")
                    })
            
            return {
                "status": "ok",
                "subscriptions": subscriptions,
                "count": len(subscriptions)
            }
        else:
            return {
                "status": "ok",
                "subscriptions": [],
                "count": 0
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to detect subscriptions: {str(e)}"
        )


@router.get("/alerts")
def get_alerts(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get spending alerts and warnings.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        List of alerts
    """
    try:
        # Get recent transactions (last 7 days)
        checks = get_prepay_checks(user_id=user_id, limit=1000)
        
        alerts = []
        week_ago = datetime.now() - timedelta(days=7)
        
        # Analyze for alerts
        blocked_count = 0
        warned_count = 0
        high_amount_total = 0.0
        unverified_merchants = set()
        
        for check in checks:
            check_date = datetime.fromtimestamp(check[10])
            
            if check_date < week_ago:
                continue
            
            decision = check[8]  # decision field
            amount = check[3]    # amount field
            merchant = check[2]  # merchant field
            
            if decision == "block":
                blocked_count += 1
            elif decision == "warn":
                warned_count += 1
            
            if amount > 50000:
                high_amount_total += amount
            
            # Check for unverified merchants
            import json
            analysis = json.loads(check[9])
            if analysis.get("risk_flags", {}).get("unverified_merchant"):
                unverified_merchants.add(merchant)
        
        # Generate alerts
        if blocked_count > 0:
            alerts.append({
                "type": "policy_violation",
                "severity": "high",
                "message": f"{blocked_count} transaction(s) blocked by policy in the last 7 days",
                "count": blocked_count
            })
        
        if warned_count > 5:
            alerts.append({
                "type": "high_warnings",
                "severity": "medium",
                "message": f"{warned_count} transactions triggered warnings",
                "count": warned_count
            })
        
        if high_amount_total > 100000:
            alerts.append({
                "type": "high_spending",
                "severity": "medium",
                "message": f"High-value transactions totaling ₹{high_amount_total:,.0f}",
                "amount": high_amount_total
            })
        
        if len(unverified_merchants) > 3:
            alerts.append({
                "type": "unverified_merchants",
                "severity": "low",
                "message": f"{len(unverified_merchants)} unverified merchants detected",
                "merchants": list(unverified_merchants)[:5]
            })
        
        return {
            "status": "ok",
            "alerts": alerts,
            "count": len(alerts)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate alerts: {str(e)}"
        )
