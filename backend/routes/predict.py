"""
Prediction endpoint - handles transaction categorization requests.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.model_adapter import ModelAdapter

router = APIRouter()

# Initialize model adapter once at module load
try:
    adapter = ModelAdapter()
    print("Model adapter initialized successfully")
except Exception as e:
    print(f"Warning: Model adapter initialization failed: {e}")
    adapter = None


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    text: str = Field(..., description="Transaction description text", min_length=1)
    meta: Optional[Dict[str, Any]] = Field(None, description="Optional metadata (MCC, time, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "STARBCKS #1023 MUMBAI 12:32PM",
                "meta": {"mcc": None, "time": "12:32PM"}
            }
        }


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    category: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Confidence score (0-1)")
    explanation: Dict[str, Any] = Field(..., description="Explanation with rule hits and top tokens")
    model_used: str = Field(..., description="Which model(s) were used")


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict the category for a transaction.
    
    Args:
        req: PredictRequest with text and optional metadata
        
    Returns:
        PredictResponse with category, confidence, and explanation
        
    Raises:
        HTTPException: If prediction fails
    """
    if adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Model adapter not initialized. Please check model files."
        )
    
    try:
        # Call model adapter
        fused = adapter.predict(req.text, req.meta)
        
        # Format response
        return PredictResponse(
            category=fused["label"],
            confidence=fused["confidence"],
            explanation=fused["rationale"],
            model_used=fused.get("model_used", "unknown")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model-status")
def get_model_status() -> Dict[str, Any]:
    """
    Get the status of loaded models.
    
    Returns:
        Dictionary with model availability information
    """
    if adapter is None:
        return {
            "status": "error",
            "message": "Model adapter not initialized",
            "models": {}
        }
    
    try:
        status = adapter.get_model_status()
        return {
            "status": "ok",
            "models": status,
            "message": "Model status retrieved successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "models": {}
        }
