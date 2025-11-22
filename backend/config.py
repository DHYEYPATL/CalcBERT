from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    
    
    
    LOCAL_ONLY: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    
    
    DB_URL: str = "sqlite+aiosqlite:///./backend/backend_feedback.db"
    
    
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    
    
    TFIDF_MODEL_DIR: str = "./saved_models/tfidf"
    DISTILBERT_DIR: str = "./saved_models/distilbert"
    
    
    RETRAIN_SYNC: bool = True 
    
    
    API_TITLE: str = "CalcBERT Backend"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Offline hybrid rule+ML transaction categorizer"
    
    class Config:
        env_file = ".env"
        case_sensitive = True



settings = Settings()
