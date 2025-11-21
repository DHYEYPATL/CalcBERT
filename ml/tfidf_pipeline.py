from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np
from scipy.special import expit  # sigmoid

class TfidfPipeline:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.clf = SGDClassifier(loss="log_loss", max_iter=1000)
        self.le = LabelEncoder()
        self._is_fitted = False

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        y = self.le.fit_transform(labels)
        self.clf.fit(X, y)
        self._is_fitted = True

    def _get_probs(self, X):
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(X)
        if hasattr(self.clf, "decision_function"):
            df = self.clf.decision_function(X)
            probs = expit(df)
            if probs.ndim == 1:
                probs = np.vstack([1-probs, probs]).T
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs
        n = len(self.le.classes_)
        return np.ones((X.shape[0], n)) / n

    def predict(self, texts):
        if not self._is_fitted:
            raise ValueError("Model not fitted or loaded.")
        X = self.vectorizer.transform(texts)
        probs = self._get_probs(X)
        preds = self.clf.predict(X)

        results = []
        for i, p in enumerate(preds):
            label = self.le.inverse_transform([p])[0]
            prob = float(probs[i].max())
            probs_map = {
                self.le.inverse_transform([j])[0]: float(probs[i][j])
                for j in range(len(self.le.classes_))
            }
            results.append({
                "label": label,
                "confidence": prob,
                "probs": probs_map,
                "top_tokens": []
            })
        return results

    def partial_fit(self, texts, labels):
        X = self.vectorizer.transform(texts)
        current = list(self.le.classes_)
        new = list(set(labels) - set(current))
        if new:
            all_classes = current + new
            self.le.fit(all_classes)
        y = self.le.transform(labels)
        classes = range(len(self.le.classes_))
        self.clf.partial_fit(X, y, classes=classes)
        self._is_fitted = True

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.vectorizer, f"{out_dir}/vectorizer.pkl")
        joblib.dump(self.clf, f"{out_dir}/model.pkl")
        joblib.dump(self.le, f"{out_dir}/label_encoder.pkl")

    def load(self, out_dir):
        self.vectorizer = joblib.load(f"{out_dir}/vectorizer.pkl")
        self.clf = joblib.load(f"{out_dir}/model.pkl")
        self.le = joblib.load(f"{out_dir}/label_encoder.pkl")
        self._is_fitted = True
