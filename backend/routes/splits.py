"""
Payment Splits API Route
Handles payment split configurations for dashboard.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.storage import save_payment_split, get_latest_payment_split, get_payment_splits_today

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
def get_latest_split(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the most recent payment split configuration.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        Latest split configuration
    """
    try:
        split_record = get_latest_payment_split(user_id=user_id)
        
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
def get_today_splits(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all payment splits created today, combined into a single view.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        Combined splits from today with aggregated totals by category
    """
    try:
        today_splits = get_payment_splits_today(user_id=user_id)
        
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