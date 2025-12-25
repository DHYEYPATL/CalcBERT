from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

import joblib
import os
import numpy as np
from scipy.special import expit
from typing import List, Optional, Dict, Any

from ml.data_pipeline import detect_low_quality_note, extract_intent_phrases
from ml.merchant_check import verify_merchant


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.65

# MUST match expected tests
BUSINESS_MERCHANT_TYPES = {"transport", "subscription", "travel"}


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class TfidfPipeline:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.clf = SGDClassifier(loss="log_loss", max_iter=1000)
        self.le = LabelEncoder()
        self._is_fitted = False

    # --------------------------------------------------------------
    # Persistence (expected by tests)
    # --------------------------------------------------------------

    @classmethod
    def load(cls, model_dir: str) -> "TfidfPipeline":
        p = cls()
        p.vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
        p.clf = joblib.load(os.path.join(model_dir, "model.pkl"))
        p.le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        p._is_fitted = True
        return p

    # --------------------------------------------------------------
    # Training (extra but safe)
    # --------------------------------------------------------------

    def fit(self, texts: List[str], labels: List[str]) -> None:
        X = self.vectorizer.fit_transform(texts)
        y = self.le.fit_transform(labels)
        self.clf.fit(X, y)
        self._is_fitted = True

    def partial_fit(self, texts: List[str], labels: List[str]) -> int:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before partial_fit.")

        filtered = [
            (t, l) for t, l in zip(texts, labels)
            if l in self.le.classes_
        ]

        if not filtered:
            return 0

        texts_f, labels_f = zip(*filtered)
        X = self.vectorizer.transform(texts_f)
        y = self.le.transform(labels_f)

        self.clf.partial_fit(X, y)
        return len(y)

    def _get_probs(self, X):
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)

        df = self.clf.decision_function(X)
        probs = expit(df)

        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T

        return probs / probs.sum(axis=1, keepdims=True)

    def predict(
        self,
        texts: List[str],
        meta: Optional[Dict[str, Any]] = None,
        merchants: Optional[List[str]] = None,   
        return_extra: bool = False               
    ) -> List[Dict[str, Any]]:

        if not self._is_fitted:
            raise ValueError("Model not fitted or loaded.")

        X = self.vectorizer.transform(texts)
        probs = self._get_probs(X)
        preds = self.clf.predict(X)

        results = []

        for i, p in enumerate(preds):
            label = self.le.inverse_transform([p])[0]
            confidence = float(probs[i].max())

            if meta and "merchant" in meta:
                merchant = meta.get("merchant")
            elif merchants and i < len(merchants):
                merchant = merchants[i]
            else:
                merchant = None

            merchant_info = verify_merchant(merchant)

            low_quality = detect_low_quality_note(texts[i])
            intents = extract_intent_phrases(texts[i])

            risk_flags = {
                "uncertain_category": confidence < CONFIDENCE_THRESHOLD,
                "personal_vs_business_conflict": (
                    label.lower().startswith("personal")
                    and merchant_info.get("merchant_type") in BUSINESS_MERCHANT_TYPES
                ),
                "unknown_merchant": not merchant_info.get("verified", False),
                "low_note_quality": low_quality,
            }

            actions = []
            if low_quality:
                actions.append("edit_note")
            if risk_flags["uncertain_category"]:
                actions.append("confirm_category")

            result = {
                "category": label,
                "confidence": round(confidence, 3),
                "splits": [],
                "risk_flags": risk_flags,
                "suggested_actions": actions,
                "top_tokens": [],
            }

            if return_extra:
                result["intent_phrases"] = intents
                result["probs"] = {
                    self.le.inverse_transform([j])[0]: float(probs[i][j])
                    for j in range(len(self.le.classes_))
                }

            results.append(result)

        return results
