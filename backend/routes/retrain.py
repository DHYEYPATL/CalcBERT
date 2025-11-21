"""
Retrain endpoint - triggers incremental model retraining.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.storage import get_feedback_samples
from backend.config import settings

router = APIRouter()


class RetrainRequest(BaseModel):
    """Request schema for retrain endpoint."""
    mode: Literal["incremental", "full"] = Field(
        default="incremental",
        description="Retrain mode: incremental or full"
    )
    model: Literal["tfidf", "distilbert"] = Field(
        default="tfidf",
        description="Model to retrain: tfidf or distilbert"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "mode": "incremental",
                "model": "tfidf"
            }
        }


class RetrainResponse(BaseModel):
    """Response schema for retrain endpoint."""
    status: str = Field(..., description="Status: started, complete, or error")
    details: str = Field(..., description="Details about the retrain operation")
    samples_used: int = Field(default=0, description="Number of feedback samples used")


def _run_incremental_tfidf() -> dict:
    """
    Run incremental TF-IDF retraining.
    
    Returns:
        Dictionary with retrain results
    """
    try:
        from ml.tfidf_pipeline import TfidfPipeline
        from ml.feedback_handler import apply_incremental_update
        
        # Load current pipeline
        pipeline = TfidfPipeline()
        pipeline.load(settings.TFIDF_MODEL_DIR)
        
        # Get feedback samples
        samples = get_feedback_samples()
        
        if not samples:
            return {
                "status": "skipped",
                "details": "No feedback samples available for retraining",
                "samples_used": 0
            }
        
        # Convert to format expected by feedback handler
        feedback_data = [(text, label) for _, text, label in samples]
        
        # Apply incremental update
        apply_incremental_update(pipeline, feedback_data)
        
        return {
            "status": "complete",
            "details": f"Incremental TF-IDF retrain completed successfully",
            "samples_used": len(samples)
        }
    except ImportError as e:
        return {
            "status": "error",
            "details": f"ML modules not available: {str(e)}",
            "samples_used": 0
        }
    except Exception as e:
        return {
            "status": "error",
            "details": f"Retrain failed: {str(e)}",
            "samples_used": 0
        }


@router.post("/retrain", response_model=RetrainResponse)
def retrain(req: RetrainRequest, background_tasks: BackgroundTasks) -> RetrainResponse:
    """
    Trigger model retraining.
    
    Args:
        req: RetrainRequest with mode and model selection
        background_tasks: FastAPI background tasks
        
    Returns:
        RetrainResponse with status and details
        
    Raises:
        HTTPException: If retrain request is invalid
    """
    # For hackathon, only support TF-IDF incremental retrain
    if req.model != "tfidf":
        raise HTTPException(
            status_code=400,
            detail="Only TF-IDF incremental retrain is supported via API. "
                   "DistilBERT retraining should be done via Colab notebook."
        )
    
    if req.mode != "incremental":
        raise HTTPException(
            status_code=400,
            detail="Only incremental mode is supported for API-triggered retraining."
        )
    
    # Run synchronously or in background based on config
    if settings.RETRAIN_SYNC:
        try:
            result = _run_incremental_tfidf()
            return RetrainResponse(**result)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Retrain failed: {str(e)}"
            )
    else:
        # Run in background
        background_tasks.add_task(_run_incremental_tfidf)
        return RetrainResponse(
            status="started",
            details="Incremental TF-IDF retrain started in background",
            samples_used=0  # Will be updated when complete
        )


@router.get("/retrain/status")
def get_retrain_status() -> dict:
    """
    Get retrain configuration and status.
    
    Returns:
        Dictionary with retrain configuration
    """
    return {
        "sync_mode": settings.RETRAIN_SYNC,
        "supported_models": ["tfidf"],
        "supported_modes": ["incremental"],
        "message": "Retrain endpoint is ready"
    }
