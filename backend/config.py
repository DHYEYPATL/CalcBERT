"""
Backend configuration and environment settings.
Provides centralized configuration for the FastAPI backend.
"""

from typing import List
from pydantic import Field
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
    API_VERSION: str = "2.0.0"
    API_DESCRIPTION: str = "Offline hybrid rule+ML transaction categorizer with prepay policy engine"
    
    # v2 Backend: Pinecone Configuration
    PINECONE_API_KEY: str = ""
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "calcbert-expenses"
    
    # v2 Backend: Vector Store Configuration
    VECTOR_DIMENSION: int = 384  # DistilBERT embedding dimension
    VECTOR_METRIC: str = "cosine"
    VECTOR_FALLBACK_PATH: str = "./backend/vector_store_fallback.pkl"
    USE_PINECONE: bool = False  # Auto-detected based on API key
    
    # v2 Backend: LangGraph Workflow Configuration
    LANGGRAPH_ENABLED: bool = False  # Set True if LangGraph is installed
    WORKFLOW_MODE: str = "sequential"  # 'sequential' or 'langgraph'
    
    # v2 Backend: Policy Engine Configuration
    EMPLOYEE_ALCOHOL_BLOCK: bool = True
    EMPLOYEE_HIGH_AMOUNT_THRESHOLD: float = 50000.0
    UNIVERSAL_BLOCK_THRESHOLD: float = 200000.0
    UNVERIFIED_MERCHANT_WARN_THRESHOLD: float = 10000.0
    
    # Mock Mode Configuration
    CALCBERT_MODE: str = Field(default="production", env="CALCBERT_MODE")  # "mock" or "production"
    ENABLE_MOCK_FALLBACK: bool = True  # Fallback to mock if models fail
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Auto-detect Pinecone availability
if settings.PINECONE_API_KEY:
    settings.USE_PINECONE = True

