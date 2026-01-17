# CalcBERT v2 Pipeline Overview

## System Architecture

CalcBERT v2 is a pre-payment risk and spend intelligence system with role-based policy enforcement and agentic AI orchestration.

### High-Level Flow

```
User Transaction → Prepay Check API → Workflow Orchestrator → Policy Engine → Decision
                                    ↓
                            ML Models (TF-IDF, DistilBERT, Rules)
                                    ↓
                            Merchant Verification & Subscription Detection
                                    ↓
                            Fusion & Explanation Builder
```

---

## Decision Matrix

This matrix defines how the system makes decisions based on various conditions:

| Condition | Decision | UI Action | Policy Reason |
|-----------|----------|-----------|---------------|
| **Confidence < 0.65** | `warn` | Edit Note, Continue Anyway, Cancel | LOW_CONFIDENCE |
| **Alcohol + Employee** | `block` | Cancel, Request Override | EMPLOYEE_ALCOHOL_RESTRICT |
| **Unknown Merchant + High Amount (>₹10k)** | `warn` | Continue Anyway, Edit Note, Cancel | UNVERIFIED_MERCHANT_HIGH_AMOUNT |
| **Amount > ₹50k (Employee)** | `warn` | Continue Anyway, Edit Note, Cancel | HIGH_AMOUNT_EMPLOYEE |
| **Amount > ₹2L (All Roles)** | `block` | Cancel, Request Override | AMOUNT_EXCEEDS_LIMIT |
| **Subscription Detected (No Approval)** | `warn` | Manage Subscription, Continue, Cancel | SUBSCRIPTION_DETECTED_NO_APPROVAL |
| **Low Quality Note** | `warn` | Edit Note, Continue Anyway, Cancel | LOW_QUALITY_NOTE |
| **Manager + Alcohol** | `allow_with_note` | Add Note and Continue, Cancel | ALCOHOL_MANAGER_NOTE_REQUIRED |
| **Normal Transaction** | `allow` | Continue | - |

---

## Component Responsibilities

### 1. ML Layer (`ml/`)

#### **Neha's Components:**
- `data_pipeline.py` - Text normalization, quality detection, intent extraction
- `tfidf_pipeline.py` - TF-IDF categorization with risk flags
- `rules.py` - Keyword-based rule matching
- `merchant_check.py` - Merchant verification using registry
- `merchant_registry.json` - 130+ verified merchants
- `subscription_detector.py` - Recurring payment detection
- `summary_builder.py` - Daily spend summaries

#### **Adya's Components:**
- `distilbert_model.py` - Deep learning categorization
- `fusion.py` - Decision fusion from multiple models
- `explain.py` - Explainability layer

### 2. Backend Layer (`backend/`)

#### **Dhyey's Components:**
- `routes/prepay.py` - Prepay check API endpoints
- `routes/summary.py` - Summary, subscription, and alerts APIs
- `policy_engine.py` - Role-based policy enforcement
- `langgraph_workflow.py` - Workflow orchestration (6 nodes)
- `vector_store.py` - Pinecone vector memory with sklearn fallback
- `storage.py` - SQLite database with prepay_checks table
- `model_adapter.py` - ML model integration layer

---

## Workflow Orchestration

### Sequential Mode (Default)

```
1. Merchant Check Node
   ├─ Verify merchant (verified/known/unknown)
   ├─ Detect subscription patterns
   └─ Check note quality

2. TF-IDF Prediction Node
   ├─ Category prediction
   ├─ Confidence scoring
   └─ Rule-based matching

3. DistilBERT Prediction Node (Optional)
   ├─ Deep learning prediction
   └─ Embedding generation

4. Policy Engine Node
   ├─ Apply role-based rules
   ├─ Check amount thresholds
   └─ Determine action (allow/warn/block)

5. Fusion Node
   ├─ Combine predictions
   ├─ Select best category
   └─ Calculate final confidence

6. Explanation Builder Node
   ├─ Generate category explanation
   ├─ Format policy reasons
   └─ Provide actionable suggestions
```

---

## API Endpoints

### Core Endpoints

| Endpoint | Method | Purpose | Owner |
|----------|--------|---------|-------|
| `/predict` | POST | Transaction categorization | Existing |
| `/feedback` | POST | User feedback collection | Existing |
| `/retrain` | POST | Model retraining trigger | Existing |
| `/prepay/check` | POST | Pre-payment approval check | Dhyey |
| `/prepay/history` | GET | Prepay check history | Dhyey |
| `/prepay/stats` | GET | Prepay statistics | Dhyey |
| `/summary/today` | GET | Daily spend summary | Dhyey |
| `/summary/subscriptions` | GET | Detected subscriptions | Dhyey |
| `/summary/alerts` | GET | Spending alerts | Dhyey |

---

## Data Flow Example

### Example: Employee Dinner Transaction

**Input:**
```json
{
  "merchant": "Swiggy",
  "amount": 1200.0,
  "note": "client meeting + dinner",
  "user_role": "employee"
}
```

**Processing:**
1. **Merchant Check**: Swiggy → verified, food_delivery
2. **Note Quality**: "client meeting + dinner" → acceptable
3. **TF-IDF**: "Restaurant & Dining" (0.85 confidence)
4. **Rules**: Matches "swiggy" → "Restaurant & Dining" (0.95 confidence)
5. **Policy**: Employee + Normal amount + Verified merchant → allow
6. **Fusion**: Rule wins (higher confidence) → "Restaurant & Dining"

**Output:**
```json
{
  "decision": "allow",
  "options": ["Continue"],
  "analysis": {
    "final_category": "Restaurant & Dining",
    "final_confidence": 0.95,
    "risk_flags": { "all": false }
  }
}
```

---

## Configuration Flags

### Feature Toggles (`backend/config.py`)

```python
# Model Availability
ENABLE_BERT = True
ENABLE_VECTOR_STORE = False
ENABLE_LANGGRAPH = True

# Mock Mode
CALCBERT_MODE = "production"  # or "mock"
ENABLE_MOCK_FALLBACK = True

# Policy Thresholds
EMPLOYEE_HIGH_AMOUNT_THRESHOLD = 50000.0
UNIVERSAL_BLOCK_THRESHOLD = 200000.0
UNVERIFIED_MERCHANT_WARN_THRESHOLD = 10000.0
```

---

## Database Schema

### `prepay_checks` Table

```sql
CREATE TABLE prepay_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount REAL NOT NULL,
    note TEXT,
    upi_id TEXT,
    user_role TEXT NOT NULL,
    date TEXT NOT NULL,
    decision TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
```

---

## Error Handling & Fallbacks

### Graceful Degradation

1. **Model Unavailable**: Falls back to rule-based classification
2. **Pinecone Unavailable**: Uses local sklearn vector store
3. **LangGraph Unavailable**: Uses sequential workflow mode
4. **ML Error**: Returns safe fallback response with `warn` decision

### Fail-Soft Response

```python
FALLBACK_SAFE_RESPONSE = {
    "decision": "warn",
    "reason": "SYSTEM_CONFIDENCE_LOW",
    "options": ["Edit Note", "Cancel"],
    "analysis": {}
}
```

---

## Testing Strategy

### Unit Tests (51 total)
- `test_policy_engine.py` (13 tests)
- `test_langgraph_workflow.py` (13 tests)
- `test_vector_store.py` (9 tests)
- `test_prepay_api.py` (16 tests)

### Integration Tests
- End-to-end prepay check flow
- API contract validation
- Database persistence verification

---

## Performance Considerations

- **TF-IDF**: ~10ms per prediction
- **DistilBERT**: ~100ms per prediction (optional)
- **Policy Engine**: <1ms (deterministic)
- **Total Latency**: <200ms for full workflow

---

## Security & Privacy

- Role-based access control (employee/manager/admin)
- No sensitive data in logs
- SQLite for local development, PostgreSQL for production
- API key required for Pinecone (optional)

---

## Future Enhancements

1. Real-time fraud detection
2. Advanced split detection with ML
3. Personalized spending insights
4. Integration with accounting systems
5. Multi-currency support
