# ml/adapter.py
# -----------------------------
# Backend-facing interface for prediction and incremental training.

from ml.tfidf_pipeline import TfidfPipeline
from ml.feedback_handler import ingest_feedback, apply_incremental_update
from ml.data_pipeline import normalize_text

MODEL_DIR = "saved_models/tfidf"
FEEDBACK_PATH = "data/feedback.json"

# Cache model so it loads only once
_model = None


def load_model():
    """Load the TF-IDF model once and keep it cached."""
    global _model
    if _model is None:
        _model = TfidfPipeline()
        _model.load(MODEL_DIR)
    return _model


def predict_text(text: str):
    """Normalize text → run prediction → return model output dict."""
    model = load_model()
    cleaned = normalize_text(text)
    result = model.predict([cleaned])[0]
    return result


def retrain_from_feedback():
    """Incrementally update model using data/feedback.json."""
    model = load_model()
    samples = ingest_feedback(FEEDBACK_PATH)
    updated_count = apply_incremental_update(
        pipeline=model,
        samples=samples,
        save_dir=MODEL_DIR
    )
    return {
        "updated_samples": updated_count,
        "status": "ok" if updated_count > 0 else "no_updates"
    }
