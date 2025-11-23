# 🔄 CalcBERT Pipeline Overview

## 📊 High-Level Architecture

Your CalcBERT project implements a *hybrid transaction categorization system* that combines:
- *Rule-based classification* (keyword matching)
- *Machine Learning models* (TF-IDF + optional DistilBERT)
- *Incremental learning* (feedback-driven model improvement)

---

## 🌊 Data Flow Pipeline


┌─────────────────────────────────────────────────────────────────┐
│                    USER REQUEST (Transaction Text)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (app.py)                      │
│                    Port: 8000                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PREDICTION ROUTE (/predict)                         │
│              File: backend/routes/predict.py                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL ADAPTER (model_adapter.py)                    │
│              Orchestrates all prediction logic                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
    ┌───────────────────┐   ┌───────────────────┐
    │  RULE-BASED       │   │  ML MODELS        │
    │  (ml/rules.py)    │   │  (ml/tfidf_       │
    │                   │   │   pipeline.py)    │
    │  - Keyword match  │   │                   │
    │  - High conf      │   │  - TF-IDF         │
    │    (0.95)         │   │  - DistilBERT     │
    └─────────┬─────────┘   └─────────┬─────────┘
              │                       │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  FUSION MODULE        │
              │  (ml/fusion.py)       │
              │                       │
              │  Combines outputs     │
              │  Rule > 0.9 → Rule    │
              │  Else → ML            │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  PREDICTION RESULT    │
              │  {                    │
              │    label,             │
              │    confidence,        │
              │    rationale,         │
              │    model_used         │
              │  }                    │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  RETURN TO USER       │
              └───────────────────────┘


---

## 📁 File Structure & Responsibilities

### *Backend Layer* (backend/)

#### 1. *app.py* - Main Application Entry Point
- *Purpose*: FastAPI application initialization
- *Key Functions*:
  - Initialize database on startup
  - Configure CORS for frontend access
  - Mount all API routes
  - Provide health check and metrics endpoints

#### 2. *model_adapter.py* - Model Orchestration Hub
- *Purpose*: Unified interface for all ML models
- *Key Functions*:
  - Load TF-IDF, DistilBERT, rules, and fusion modules
  - predict(): Main prediction method that combines all models
  - get_model_status(): Check which models are loaded
- *Flow*:
  1. Apply rules → get rule output (if match)
  2. Apply ML model → get ML output
  3. Fuse both using fusion logic
  4. Return combined prediction

#### 3. *storage.py* - Database Management
- *Purpose*: SQLite database for feedback storage
- *Key Functions*:
  - init_db(): Create feedback table
  - save_feedback(): Store user corrections
  - get_feedback_samples(): Retrieve feedback for retraining
  - get_feedback_count(): Count total feedback entries
  - get_recent_feedback(): Get recent feedback by time window

#### 4. *config.py* - Configuration Settings
- *Purpose*: Centralized configuration
- *Contains*:
  - API settings (host, port, title)
  - Model paths (TF-IDF, DistilBERT directories)
  - CORS settings
  - Retrain settings

### *Routes Layer* (backend/routes/)

#### 1. *predict.py* - Prediction Endpoint
- *Endpoint*: POST /predict
- *Input*: {text: str, meta: dict}
- *Output*: {category, confidence, explanation, model_used}
- *Process*:
  1. Validate request
  2. Call model_adapter.predict()
  3. Format and return response

#### 2. *feedback.py* - Feedback Collection
- *Endpoints*:
  - POST /feedback: Save user correction
  - GET /feedback/count: Get feedback statistics
- *Purpose*: Collect user corrections for incremental learning
- *Storage*: Saves to SQLite via storage.py

#### 3. *retrain.py* - Model Retraining
- *Endpoints*:
  - POST /retrain: Trigger incremental retraining
  - GET /retrain/status: Check retrain configuration
- *Process*:
  1. Fetch feedback samples from database
  2. Load current TF-IDF model
  3. Apply incremental update using partial_fit()
  4. Save updated model

---

### *ML Layer* (ml/)

#### 1. *rules.py* - Rule-Based Classification
- *Purpose*: High-confidence keyword matching
- *Categories Covered*:
  - Coffee & Beverages
  - Transportation
  - Restaurant & Dining
  - Online Shopping
  - Groceries
  - Entertainment
  - Gas & Fuel
  - Healthcare
- *Output*: {label, confidence: 0.95, matches: [...], source: "rule-based"}

#### 2. *tfidf_pipeline.py* - TF-IDF ML Model
- *Purpose*: Machine learning classification using TF-IDF + SGDClassifier
- *Key Methods*:
  - fit(): Initial training
  - predict(): Predict with confidence scores
  - partial_fit(): Incremental learning from feedback
  - save()/load(): Model persistence
- *Output*: {label, confidence, probs, top_tokens}

#### 3. *fusion.py* - Prediction Fusion Logic
- *Purpose*: Combine rule-based and ML predictions
- *Strategy*:
  - If rule confidence ≥ 0.9 → Use rule prediction
  - Otherwise → Use ML prediction
- *Output*: Combined prediction with rationale

#### 4. *distilbert_model.py* - Deep Learning Model (Optional)
- *Purpose*: Advanced transformer-based classification
- *Status*: Optional, falls back to TF-IDF if not available

#### 5. *feedback_handler.py* - Incremental Learning
- *Purpose*: Apply feedback to update models
- *Function*: apply_incremental_update(pipeline, feedback_data)

#### 6. *generate_alias.py* - Text Normalization
- *Purpose*: Generate aliases for transaction text normalization
- *Use*: Preprocessing step for better matching

---

## 🔄 Complete Request-Response Flow

### *Scenario 1: Prediction Request*


1. USER sends POST /predict
   {
     "text": "STARBUCKS #1023 MUMBAI 12:32PM",
     "meta": {"mcc": null, "time": "12:32PM"}
   }

2. predict.py receives request
   ↓
3. Calls model_adapter.predict(text, meta)
   ↓
4. model_adapter.py orchestrates:
   
   a) rules.py checks for keyword matches
      → Finds "starbucks" pattern
      → Returns: {
          label: "Coffee & Beverages",
          confidence: 0.95,
          matches: ["starbucks"]
        }
   
   b) tfidf_pipeline.py predicts
      → Vectorizes text
      → Classifies using SGDClassifier
      → Returns: {
          label: "Coffee & Beverages",
          confidence: 0.87,
          top_tokens: ["starbucks", "coffee"]
        }
   
   c) fusion.py combines outputs
      → Rule confidence (0.95) > 0.9
      → Uses rule prediction
      → Returns: {
          label: "Coffee & Beverages",
          confidence: 0.95,
          rationale: {
            rule_hits: ["starbucks"],
            top_tokens: ["starbucks", "coffee"]
          },
          model_used: "rule"
        }

5. predict.py formats response
   ↓
6. Returns to USER:
   {
     "category": "Coffee & Beverages",
     "confidence": 0.95,
     "explanation": {...},
     "model_used": "rule"
   }


### *Scenario 2: Feedback & Retraining*


1. USER provides feedback (correction)
   POST /feedback
   {
     "text": "UNKNOWN CAFE DELHI",
     "correct_label": "Coffee & Beverages",
     "user_id": "dhyey"
   }

2. feedback.py receives request
   ↓
3. Calls storage.save_feedback()
   ↓
4. storage.py saves to SQLite database
   → Table: feedback
   → Columns: id, text, correct_label, user_id, created_at
   ↓
5. Returns: {status: "saved", id: 123}

---

6. LATER: Admin triggers retrain
   POST /retrain
   {
     "mode": "incremental",
     "model": "tfidf"
   }

7. retrain.py receives request
   ↓
8. Calls _run_incremental_tfidf()
   ↓
9. Fetches feedback from storage.get_feedback_samples()
   ↓
10. Loads current TF-IDF model
    ↓
11. Calls pipeline.partial_fit(texts, labels)
    → Updates model weights incrementally
    ↓
12. Saves updated model
    ↓
13. Returns: {
      status: "complete",
      details: "Retrain successful",
      samples_used: 15
    }


---

## 🎯 Key Design Patterns

### 1. *Hybrid Approach*
- *Rules*: Fast, high-confidence for known patterns
- *ML*: Flexible, learns from data for unknown patterns
- *Fusion*: Best of both worlds

### 2. *Incremental Learning*
- Models update without full retraining
- Uses partial_fit() for online learning
- Feedback stored in database for batch updates

### 3. *Graceful Degradation*
- If DistilBERT unavailable → Falls back to TF-IDF
- If TF-IDF unavailable → Uses rules only
- If rules don't match → Uses ML only

### 4. *Separation of Concerns*
- *Backend*: API, routing, orchestration
- *ML*: Model logic, training, prediction
- *Storage*: Database operations
- *Routes*: Endpoint-specific logic

---

## 📊 Database Schema

### *feedback* table (SQLite)
sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,              -- Transaction description
    correct_label TEXT NOT NULL,     -- User-corrected category
    user_id TEXT,                    -- Optional user identifier
    created_at INTEGER NOT NULL      -- Unix timestamp
)


---

## 🚀 Startup Sequence


1. Run: uvicorn backend.app:app --reload --port 8000

2. app.py startup_event() executes:
   ├─ init_db() → Creates/verifies SQLite database
   ├─ Loads configuration from config.py
   └─ Mounts routes (predict, feedback, retrain)

3. predict.py module loads:
   └─ Initializes ModelAdapter
      ├─ Loads TF-IDF from saved_models/tfidf/
      ├─ Attempts to load DistilBERT (optional)
      ├─ Loads rules module
      └─ Loads fusion module

4. Server ready at http://localhost:8000
   ├─ /docs → API documentation
   ├─ /predict → Prediction endpoint
   ├─ /feedback → Feedback collection
   └─ /retrain → Model retraining


---

## 🔍 Model Decision Logic (Fusion)

python
if rule_output and rule_output.confidence >= 0.9:
    # High-confidence rule match
    return rule_output
else:
    # Use ML prediction
    if distilbert_available:
        return distilbert_output
    elif tfidf_available:
        return tfidf_output
    else:
        raise Error("No models available")


---

## 📈 Incremental Learning Flow


User Feedback → SQLite Database → Periodic Retrain
                                         ↓
                                   partial_fit()
                                         ↓
                                   Updated Model
                                         ↓
                                   Better Predictions


---

## 🎓 Summary

Your pipeline implements a *production-ready, offline-first transaction categorization system* with:

✅ *Multi-model architecture* (Rules + TF-IDF + DistilBERT)  
✅ *Intelligent fusion* (Combines strengths of each model)  
✅ *Incremental learning* (Improves from user feedback)  
✅ *Graceful fallbacks* (Works even if some models fail)  
✅ *Clean separation* (Backend, ML, Storage layers)  
✅ *RESTful API* (Easy integration with UI/frontend)  
✅ *Persistent storage* (SQLite for feedback)  
✅ *Explainable predictions* (Rationale with rule hits and top tokens)

The architecture is *modular, **scalable, and **hackathon-optimized* for rapid iteration!