# CalcBERT Backend API

FastAPI backend for CalcBERT - an offline hybrid rule+ML transaction categorizer with incremental learning.

## Quick Start

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# From project root
cd backend
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **Base URL**: http://127.0.0.1:8000
- **API Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## API Endpoints

### 1. Root & System Endpoints

#### GET `/`
Get API information and available endpoints.

```bash
curl http://127.0.0.1:8000/
```

#### GET `/health`
Health check endpoint.

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "CalcBERT Backend",
  "version": "1.0.0"
}
```

#### GET `/metrics`
Get model performance metrics.

```bash
curl http://127.0.0.1:8000/metrics
```

### 2. Prediction Endpoints

#### POST `/predict`
Predict category for a transaction.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "STARBCKS #1023 MUMBAI 12:32PM",
    "meta": {"mcc": null, "time": "12:32PM"}
  }'
```

**Response:**
```json
{
  "category": "Coffee & Beverages",
  "confidence": 0.94,
  "explanation": {
    "rule_hits": ["starbucks"],
    "top_tokens": [
      {"token": "starbucks", "score": 0.45},
      {"token": "coffee", "score": 0.12}
    ]
  },
  "model_used": "fusion"
}
```

#### GET `/model-status`
Get status of loaded models.

```bash
curl http://127.0.0.1:8000/model-status
```

**Response:**
```json
{
  "status": "ok",
  "models": {
    "tfidf": true,
    "distilbert": false,
    "rules": true,
    "fusion": true
  },
  "message": "Model status retrieved successfully"
}
```

### 3. Feedback Endpoints

#### POST `/feedback`
Submit user correction for a transaction.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "STARBCKS #1023 MUMBAI",
    "correct_label": "Restaurant & Dining",
    "user_id": "dhyey"
  }'
```

**Response:**
```json
{
  "status": "saved",
  "id": 42,
  "message": "Feedback saved successfully with ID 42"
}
```

#### GET `/feedback/count`
Get total feedback count.

```bash
curl http://127.0.0.1:8000/feedback/count
```

**Response:**
```json
{
  "status": "ok",
  "total_feedback": 42,
  "message": "Total feedback samples: 42"
}
```

### 4. Retrain Endpoints

#### POST `/retrain`
Trigger incremental model retraining.

**Request:**
```bash
curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "incremental",
    "model": "tfidf"
  }'
```

**Response:**
```json
{
  "status": "complete",
  "details": "Incremental TF-IDF retrain completed successfully",
  "samples_used": 42
}
```

#### GET `/retrain/status`
Get retrain configuration.

```bash
curl http://127.0.0.1:8000/retrain/status
```

## Complete Workflow Example

```bash
# 1. Check health
curl http://127.0.0.1:8000/health

# 2. Predict a transaction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "AMAZON.COM PURCHASE"}'

# 3. Submit correction if prediction was wrong
curl -X POST http://127.0.0.1:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "text": "AMAZON.COM PURCHASE",
    "correct_label": "Online Shopping",
    "user_id": "user123"
  }'

# 4. Check feedback count
curl http://127.0.0.1:8000/feedback/count

# 5. Trigger incremental retrain
curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"mode": "incremental", "model": "tfidf"}'

# 6. Verify improved prediction
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "AMAZON.COM PURCHASE"}'
```

## Testing

### Run Unit Tests

```bash
# From project root
pytest tests/test_api.py -v
```

### Run Performance Benchmark

```bash
# Make sure server is running first
python backend/bench_api.py
```

## Docker Deployment

### Build Image

```bash
docker build -t calcbert-backend -f backend/Dockerfile .
```

### Run Container

```bash
docker run -p 8000:8000 \
  -v $(pwd)/saved_models:/app/saved_models \
  -v $(pwd)/data:/app/data \
  calcbert-backend
```

## Configuration

Configuration is managed in `backend/config.py`. Key settings:

- `LOCAL_ONLY`: Restrict to localhost (default: True)
- `HOST`: Server host (default: 127.0.0.1)
- `PORT`: Server port (default: 8000)
- `ALLOWED_ORIGINS`: CORS allowed origins
- `TFIDF_MODEL_DIR`: Path to TF-IDF model files
- `DISTILBERT_DIR`: Path to DistilBERT model files
- `RETRAIN_SYNC`: Synchronous retrain (default: True)

Override settings using a `.env` file in the project root.

## Architecture

```
backend/
├── app.py              # Main FastAPI application
├── config.py           # Configuration settings
├── storage.py          # SQLite feedback storage
├── model_adapter.py    # ML model integration
├── routes/
│   ├── predict.py      # Prediction endpoints
│   ├── feedback.py     # Feedback endpoints
│   └── retrain.py      # Retrain endpoints
├── bench_api.py        # Performance benchmarking
└── Dockerfile          # Container configuration
```

## Database Schema

**Feedback Table:**
```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    correct_label TEXT NOT NULL,
    user_id TEXT,
    created_at INTEGER NOT NULL
);
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid parameters)
- `422`: Validation error (invalid input schema)
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)

Error responses include a `detail` field with error description.

## Development

### Code Style

```bash
# Format code
black backend/

# Lint code
flake8 backend/
```

### Adding New Endpoints

1. Create route file in `backend/routes/`
2. Define request/response schemas with Pydantic
3. Implement endpoint logic
4. Register router in `backend/app.py`
5. Add tests in `tests/test_api.py`

## Troubleshooting

### Models Not Loading

Check that model files exist:
```bash
ls -la saved_models/tfidf/
```

View model status:
```bash
curl http://127.0.0.1:8000/model-status
```

### Database Issues

Initialize database manually:
```python
from backend.storage import init_db
init_db()
```

### CORS Errors

Add your frontend URL to `ALLOWED_ORIGINS` in `backend/config.py`.

## License

Part of the CalcBERT project.
