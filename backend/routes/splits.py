"""
Payment Splits API Route
Handles payment split configurations for dashboard.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.storage import save_payment_split, get_latest_payment_split, get_payment_splits_today
from backend.storage import get_prepay_checks, save_prepay_check
router = APIRouter()


class SplitItem(BaseModel):
    """Individual split item."""
    label: str
    amount: float


class SaveSplitRequest(BaseModel):
    """Request schema for saving splits."""
    user_id: str = Field(..., description="User identifier")
    total_amount: float = Field(..., description="Total amount", gt=0)
    splits: List[SplitItem] = Field(..., description="List of split items")


@router.post("")
def save_split(req: SaveSplitRequest) -> Dict[str, Any]:
    """
    Save a payment split configuration to the database.
    
    Args:
        req: SaveSplitRequest with user_id, total_amount, and splits
        
    Returns:
        Response with saved split ID
    """
    try:
        print(f"[splits] Received split save request: user_id={req.user_id}, total_amount={req.total_amount}, splits_count={len(req.splits)}")
        splits_json = json.dumps([{"label": s.label, "amount": s.amount} for s in req.splits])
        print(f"[splits] Serialized splits JSON: {splits_json}")
        
        split_id = save_payment_split(
            user_id=req.user_id,
            total_amount=req.total_amount,
            splits_json=splits_json
        )
        
        print(f"[splits] Split saved successfully with ID: {split_id}")
        return {
            "status": "ok",
            "split_id": split_id,
            "message": "Split saved successfully"
        }
        
    except Exception as e:
        import traceback
        print(f"[splits] Error saving split: {str(e)}")
        print(f"[splits] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save split: {str(e)}"
        )


@router.get("/latest")
def get_latest_split(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get the most recent payment split configuration.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        Latest split configuration
    """
    try:
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        split_record = get_latest_payment_split(user_id=effective_user_id)
        
        if not split_record:
            return {
                "status": "ok",
                "split": None,
                "message": "No splits found"
            }
        
        # Parse JSON
        splits = json.loads(split_record[3])  # splits_json field
        
        return {
            "status": "ok",
            "split": {
                "id": split_record[0],
                "user_id": split_record[1],
                "total_amount": split_record[2],
                "splits": splits,
                "created_at": split_record[4]
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest split: {str(e)}"
        )


@router.get("/today")
def get_today_splits(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid")  # Backward compatibility
) -> Dict[str, Any]:
    """
    Get all payment splits created recently (last 7 days), combined into a single view.
    Falls back to showing recent data if no splits from today.
    
    Args:
        user_id: Optional filter by user ID
        userid: Optional filter by user ID (backward compatibility)
        
    Returns:
        Combined splits from recent period with aggregated totals by category
    """
    try:
        # Use user_id if provided, otherwise use userid for backward compatibility
        effective_user_id = user_id or userid
        
        # First try to get today's splits
        today_splits = get_payment_splits_today(user_id=effective_user_id)
        
        # If no splits today, get recent splits (last 7 days)
        if not today_splits:
            import sqlite3
            from backend.storage import DB_PATH
            
            seven_days_ago = int((datetime.now() - timedelta(days=7)).timestamp())
            
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            
            if effective_user_id:
                c.execute(
                    "SELECT * FROM payment_splits WHERE user_id = ? AND created_at >= ? ORDER BY created_at DESC LIMIT 50",
                    (effective_user_id, seven_days_ago)
                )
            else:
                c.execute(
                    "SELECT * FROM payment_splits WHERE created_at >= ? ORDER BY created_at DESC LIMIT 50",
                    (seven_days_ago,)
                )
            
            today_splits = c.fetchall()
            conn.close()
        
        if not today_splits:
            return {
                "status": "ok",
                "splits": [],
                "combined_splits": [],
                "total_amount": 0.0,
                "count": 0
            }
        
        # Combine all splits from today
        category_totals = {}
        all_splits = []
        total_amount = 0.0
        
        for split_record in today_splits:
            # split_record: (id, user_id, total_amount, splits_json, created_at)
            splits = json.loads(split_record[3])  # splits_json field
            total_amount += float(split_record[2])  # total_amount
            
            all_splits.append({
                "id": split_record[0],
                "total_amount": float(split_record[2]),
                "splits": splits,
                "created_at": split_record[4]
            })
            
            # Aggregate by category
            for split in splits:
                label = split.get("label", "Unknown")
                amount = split.get("amount", 0)
                category_totals[label] = category_totals.get(label, 0) + float(amount)
        
        # Format combined splits for dashboard
        combined_splits = [
            {"label": cat, "amount": float(total)}
            for cat, total in category_totals.items()
        ]
        
        return {
            "status": "ok",
            "splits": all_splits,
            "combined_splits": combined_splits,
            "total_amount": float(total_amount),
            "count": len(today_splits)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get today's splits: {str(e)}"
        )






@router.get("/transactions")
def get_transactions(
    user_id: Optional[str] = Query(None, alias="user_id"),
    userid: Optional[str] = Query(None, alias="userid"),
    range: str = Query("monthly", alias="range")  # daily=1, weekly=7, monthly=30
) -> Dict[str, Any]:
    """Get transactions (prepay_checks) from DB. Optional range: daily, weekly, monthly."""
    effective_user_id = user_id or userid
    days = 1 if range == "daily" else (7 if range == "weekly" else 30)
    cutoff_ts = int((datetime.now() - timedelta(days=days)).timestamp())
    
    checks = get_prepay_checks(user_id=effective_user_id, limit=2000)
    transactions = []
    for check in checks:
        created_at_ts = check[10]
        if created_at_ts < cutoff_ts:
            continue
        check_id, uid, merchant, amount, note, upi_id, user_role, date_str, decision, analysis_json, _ = check
        try:
            analysis = json.loads(analysis_json) if analysis_json else {}
        except Exception:
            analysis = {}
        transactions.append({
            "id": check_id, "user_id": uid, "merchant": merchant, "amount": float(amount),
            "note": note or "", "upi_id": upi_id or "", "user_role": user_role,
            "date": date_str, "decision": decision, "analysis": analysis,
            "created_at": datetime.fromtimestamp(created_at_ts).isoformat()
        })
    return {"status": "ok", "transactions": transactions, "count": len(transactions), "range": range}

@router.post("/transactions/{transaction_id}/split")  # maps to /transactions/24/split
def save_transaction_split(transaction_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save split for specific transaction ID.
    """
    splits_json = json.dumps(payload["splits"])
    split_id = save_payment_split(
        user_id=payload["user_id"],
        total_amount=payload["total_amount"],
        splits_json=splits_json
    )
    return {"status": "ok", "split_id": split_id, "transaction_id": transaction_id}

