"""
Backend configuration and environment settings.
Provides centralized configuration for the FastAPI backend.
"""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Server configuration
    LOCAL_ONLY: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    # Database configuration
    DB_URL: str = "sqlite+aiosqlite:///./backend/backend_feedback.db"
    
    # CORS configuration
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    
    # Model paths
    TFIDF_MODEL_DIR: str = "./saved_models/tfidf"
    DISTILBERT_DIR: str = "./saved_models/distilbert"
    
    # Retrain configuration
    RETRAIN_SYNC: bool = True  # Set False to spawn background task
    
    # API configuration
    API_TITLE: str = "CalcBERT Backend"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Offline hybrid rule+ML transaction categorizer"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
