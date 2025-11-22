
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.storage import save_feedback, get_feedback_count

router = APIRouter()


class FeedbackRequest(BaseModel):
    
    text: str = Field(..., description="Transaction text", min_length=1)
    correct_label: str = Field(..., description="Correct category label", min_length=1)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "STARBCKS #1023 MUMBAI",
                "correct_label": "Coffee & Beverages",
                "user_id": "dhyey"
            }
        }


class FeedbackResponse(BaseModel):
    
    status: str = Field(..., description="Status of the operation")
    id: int = Field(..., description="ID of the saved feedback")
    message: str = Field(..., description="Human-readable message")


@router.post("/feedback", response_model=FeedbackResponse)
def post_feedback(req: FeedbackRequest) -> FeedbackResponse:
    
    try:
        
        fid = save_feedback(req.text, req.correct_label, req.user_id)
        
        return FeedbackResponse(
            status="saved",
            id=fid,
            message=f"Feedback saved successfully with ID {fid}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save feedback: {str(e)}"
        )


@router.get("/feedback/count")
def get_feedback_stats() -> dict:
    
    try:
        count = get_feedback_count()
        return {
            "status": "ok",
            "total_feedback": count,
            "message": f"Total feedback samples: {count}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get feedback count: {str(e)}"
        )
