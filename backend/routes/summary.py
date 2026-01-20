"""
Summary and Subscription API Routes - v2 Backend
Provides daily summaries, subscription management, and alerts.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd

from backend.storage import get_prepay_checks, get_latest_payment_split, get_user_subscriptions, save_subscription, delete_subscription, get_payment_splits_today

router = APIRouter()


@router.get("/today")
def get_daily_summary(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get daily spend summary.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        Daily summary with spend by category, confidence, etc.
    """
    try:
        from ml.summary_builder import build_daily_summary
        
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        # Get recent transactions (last 7 days)
        checks = get_prepay_checks(user_id=effective_user_id, limit=1000)
        
        # Filter for recent period (last 7 days)
        today = datetime.now().date()
        seven_days_ago = today - timedelta(days=7)
        today_checks = []
        
        for check in checks:
            # check[10] is created_at timestamp
            check_date = datetime.fromtimestamp(check[10]).date()
            # Include checks from last 7 days
            if check_date >= seven_days_ago:
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
        
        # Include ALL payment splits from recent period (last 7 days)
        if effective_user_id:
            import json
            import sqlite3
            from backend.storage import DB_PATH, get_payment_splits_today
            
            # First try today's splits
            today_splits = get_payment_splits_today(user_id=effective_user_id)
            
            # If no splits today, get recent splits (last 7 days)
            if not today_splits:
                seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp())
                
                conn = sqlite3.connect(DB_PATH)
                c = conn.cursor()
                c.execute(
                    "SELECT * FROM payment_splits WHERE user_id = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 50",
                    (effective_user_id, seven_days_ago)
                )
                today_splits = c.fetchall()
                conn.close()
            
            for split_record in today_splits:
                # Parse splits and add to today_spend_by_category
                splits = json.loads(split_record[3])  # splits_json field
                for split in splits:
                    label = split.get("label", "Unknown")
                    amount = split.get("amount", 0)
                    # Add to category totals (combining with transaction data)
                    if "today_spend_by_category" not in summary:
                        summary["today_spend_by_category"] = {}
                    summary["today_spend_by_category"][label] = summary["today_spend_by_category"].get(label, 0) + float(amount)
        
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
def get_subscriptions(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get subscriptions - includes both auto-detected from transactions AND user-added subscriptions.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        List of all subscriptions (detected + user-added)
    """
    try:
        from ml.subscription_detector import detect_subscription
        
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        subscriptions = []
        
        # 1. Get auto-detected subscriptions from transactions
        checks = get_prepay_checks(user_id=effective_user_id, limit=1000)
        
        if checks:
            df = pd.DataFrame(checks, columns=[
                "id", "user_id", "merchant", "amount", "note",
                "upi_id", "user_role", "date", "decision", "analysis_json", "created_at"
            ])
            
            # Group by merchant and check for subscriptions
            for merchant in df["merchant"].unique():
                merchant_df = df[df["merchant"] == merchant]
                
                # Use most common amount for this merchant
                common_amount = merchant_df["amount"].mode()[0] if len(merchant_df) > 0 else 0
                
                # Detect subscription
                result = detect_subscription(df, merchant, common_amount)
                
                if result["is_subscription"]:
                    subscriptions.append({
                        "id": f"detected_{merchant}",
                        "merchant": merchant,
                        "name": merchant,
                        "amount": float(common_amount),
                        "period": result.get("period", "monthly"),
                        "count": result.get("count", 0),
                        "reason": result.get("reason", ""),
                        "source": "detected"
                    })
        
        # 2. Get user-added subscriptions from database
        if effective_user_id:
            user_subs = get_user_subscriptions(user_id=effective_user_id)
            for sub in user_subs:
                # sub: (id, user_id, name, amount, period, created_at)
                subscriptions.append({
                    "id": f"user_{sub[0]}",
                    "merchant": sub[2],  # name
                    "name": sub[2],
                    "amount": float(sub[3]),
                    "period": sub[4],  # monthly/yearly
                    "count": 0,
                    "reason": "User-added subscription",
                    "source": "user",
                    "db_id": sub[0]
                })
        
        return {
            "status": "ok",
            "subscriptions": subscriptions,
            "count": len(subscriptions)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get subscriptions: {str(e)}"
        )


class AddSubscriptionRequest(BaseModel):
    """Request schema for adding subscription."""
    user_id: str = Field(..., description="User identifier")
    name: str = Field(..., description="Subscription name")
    amount: float = Field(..., gt=0, description="Subscription amount")
    period: str = Field(default="monthly", description="Billing period (monthly, yearly)")


@router.post("/subscriptions")
def add_subscription(req: AddSubscriptionRequest) -> Dict[str, Any]:
    """
    Add a user subscription to the database.
    
    Args:
        req: AddSubscriptionRequest with user_id, name, amount, period
        
    Returns:
        Response with subscription ID
    """
    try:
        # Validate period
        period = req.period.lower()
        if period not in ["monthly", "yearly"]:
            period = "monthly"
        
        sub_id = save_subscription(
            user_id=req.user_id,
            name=req.name,
            amount=req.amount,
            period=period
        )
        
        return {
            "status": "ok",
            "subscription_id": sub_id,
            "message": "Subscription added successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add subscription: {str(e)}"
        )


@router.delete("/subscriptions/{subscription_id}")
def remove_subscription(
    subscription_id: int,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Delete a user subscription from the database.
    
    Args:
        subscription_id: Subscription ID to delete
        user_id: Optional user ID for verification
        
    Returns:
        Response with deletion status
    """
    try:
        deleted = delete_subscription(subscription_id, user_id)
        
        if deleted:
            return {
                "status": "ok",
                "message": "Subscription deleted successfully"
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Subscription not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete subscription: {str(e)}"
        )


@router.get("/alerts")
def get_alerts(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get spending alerts and warnings.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        List of alerts
    """
    try:
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        # Get recent transactions (last 7 days)
        checks = get_prepay_checks(user_id=effective_user_id, limit=1000)
        
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


@router.get("/top-merchants")
def get_top_merchants(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),  # Backward compatibility
    limit: int = 5
) -> Dict[str, Any]:
    """
    Get top merchants by transaction count from database.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        limit: Number of top merchants to return
        
    Returns:
        List of top merchants with counts and totals
    """
    try:
        from collections import Counter
        
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        checks = get_prepay_checks(user_id=effective_user_id, limit=1000)
        
        # Count merchants
        merchant_counts = Counter()
        merchant_totals = {}
        
        for check in checks:
            merchant = check[2]  # merchant field
            amount = check[3]    # amount field
            merchant_counts[merchant] += 1
            merchant_totals[merchant] = merchant_totals.get(merchant, 0) + amount
        
        # Get top merchants
        top_merchants = []
        for merchant, count in merchant_counts.most_common(limit):
            top_merchants.append({
                "merchant": merchant,
                "transaction_count": count,
                "total_amount": float(merchant_totals[merchant])
            })
        
        return {
            "status": "ok",
            "merchants": top_merchants,
            "count": len(top_merchants)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get top merchants: {str(e)}"
        )


@router.get("/corrections")
def get_corrections_history(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),  # Backward compatibility
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get manual corrections history from feedback table.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        limit: Maximum number of corrections to return
        
    Returns:
        List of corrections with original text and corrected category
    """
    try:
        from backend.storage import get_feedback_samples
        
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        feedback_samples = get_feedback_samples(limit=limit)
        
        corrections = []
        for fid, text, correct_label in feedback_samples:
            corrections.append({
                "id": fid,
                "original_text": text,
                "corrected_category": correct_label,
            })
        
        return {
            "status": "ok",
            "corrections": corrections,
            "count": len(corrections)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get corrections: {str(e)}"
        )


@router.get("/confidence-trend")
def get_confidence_trend(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),  # Backward compatibility
    days: int = 7
) -> Dict[str, Any]:
    """
    Get confidence trend over the last N days.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        days: Number of days to analyze
        
    Returns:
        Daily confidence averages for chart
    """
    try:
        from datetime import date, timedelta
        import json
        from collections import defaultdict
        
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        checks = get_prepay_checks(user_id=effective_user_id, limit=1000)
        
        # Group by date
        daily_confidences = defaultdict(list)
        today = date.today()
        
        for check in checks:
            check_date = datetime.fromtimestamp(check[10]).date()
            
            # Only include last N days
            if (today - check_date).days > days:
                continue
            
            # Extract confidence from analysis_json
            analysis_json = check[9]
            analysis = json.loads(analysis_json)
            confidence = analysis.get("final_confidence", 0.0)
            
            if confidence > 0:
                daily_confidences[str(check_date)].append(confidence)
        
        # Build trend data (one value per day)
        trend_data = []
        for i in range(days):
            day = today - timedelta(days=i)
            day_str = str(day)
            
            if day_str in daily_confidences:
                avg_confidence = sum(daily_confidences[day_str]) / len(daily_confidences[day_str])
                trend_data.append({
                    "date": day_str,
                    "confidence": round(float(avg_confidence) * 100, 1)  # Convert to percentage
                })
            else:
                trend_data.append({
                    "date": day_str,
                    "confidence": 0
                })
        
        # Sort by date ascending for chart
        trend_data.sort(key=lambda x: x["date"])
        
        return {
            "status": "ok",
            "trend": trend_data,
            "values": [d["confidence"] for d in trend_data]  # For LineChart component
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get confidence trend: {str(e)}"
        )


@router.get("/spend-by-category")
def get_spend_by_category(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),  # Backward compatibility
    range: str = Query("weekly", alias="range")  # daily=1, weekly=7, monthly=30
) -> Dict[str, Any]:
    """
    Get spend breakdown by category from DB. Supports daily, weekly, monthly.
    Includes prepay_checks (transactions) AND payment_splits. All from DB.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional (backward compat)
        range: daily (1d), weekly (7d), monthly (30d)
    """
    try:
        import json
        import sqlite3
        from backend.storage import DB_PATH
        
        effective_user_id = user_id or userid
        days = 1 if range == "daily" else (7 if range == "weekly" else 30)
        today = datetime.now().date()
        cutoff_date = today - timedelta(days=days)
        cutoff_ts = int(datetime.combine(cutoff_date, datetime.min.time()).timestamp())
        
        category_totals = {}
        
        # 1. Prepay checks (transactions) in range
        checks = get_prepay_checks(user_id=effective_user_id, limit=2000)
        for check in checks:
            if check[10] < cutoff_ts:
                continue
            try:
                analysis = json.loads(check[9]) if check[9] else {}
            except Exception:
                analysis = {}
            cat = analysis.get("final_category") or analysis.get("category") or "Unknown"
            amt = float(check[3])
            category_totals[cat] = category_totals.get(cat, 0) + amt
        
        # 2. Payment splits in range
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if effective_user_id:
            c.execute(
                "SELECT * FROM payment_splits WHERE user_id = ? AND created_at >= ? ORDER BY created_at DESC",
                (effective_user_id, cutoff_ts)
            )
        else:
            c.execute(
                "SELECT * FROM payment_splits WHERE created_at >= ? ORDER BY created_at DESC",
                (cutoff_ts,)
            )
        rows = c.fetchall()
        conn.close()
        
        for row in rows:
            try:
                spl = json.loads(row[3])
            except Exception:
                spl = []
            for s in spl:
                lbl = s.get("label", "Unknown")
                amt = float(s.get("amount", 0))
                category_totals[lbl] = category_totals.get(lbl, 0) + amt
        
        split_data = [{"label": k, "amount": float(v)} for k, v in category_totals.items()]
        total_amount = sum(category_totals.values())
        
        return {
            "status": "ok",
            "split_data": split_data,
            "total_amount": float(total_amount),
            "categories": list(category_totals.keys()),
            "range": range,
            "days": days,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get spend by category: {str(e)}")
