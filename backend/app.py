from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import settings
from backend.routes import predict, feedback, retrain
from backend.storage import init_db


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    
    print("=" * 60)
    print("CalcBERT Backend Starting...")
    print("=" * 60)
    
    
    try:
        init_db()
        print("✓ Database initialized")
    except Exception as e:
        print(f"⚠ Database initialization warning: {e}")
    
    print("=" * 60)
    print(f"Server running at http://{settings.HOST}:{settings.PORT}")
    print(f"API docs available at http://{settings.HOST}:{settings.PORT}/docs")
    print("=" * 60)


@app.on_event("shutdown")
def shutdown_event():
    
    print("CalcBERT Backend shutting down...")



app.include_router(predict.router, prefix="", tags=["Prediction"])
app.include_router(feedback.router, prefix="", tags=["Feedback"])
app.include_router(retrain.router, prefix="", tags=["Retrain"])


@app.get("/", tags=["Root"])
def root():
    
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "predict": "/predict",
            "feedback": "/feedback",
            "retrain": "/retrain",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["System"])
def health():
   
    return {
        "status": "ok",
        "service": "CalcBERT Backend",
        "version": settings.API_VERSION
    }


@app.get("/metrics", tags=["System"])
def metrics():
    
    metrics_path = "metrics/tfidf_metrics.json"
    
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            return {
                "status": "ok",
                "metrics": metrics_data,
                "source": metrics_path
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load metrics: {str(e)}",
                "metrics": None
            }
    else:
        return {
            "status": "not_found",
            "message": f"Metrics file not found at {metrics_path}",
            "metrics": None
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )
