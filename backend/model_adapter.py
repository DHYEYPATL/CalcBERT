"""
Model adapter - integrates ML models and provides unified prediction interface.
Loads TF-IDF, DistilBERT (optional), rules, and fusion modules.
"""

import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to path for ml module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import settings


class ModelAdapter:
    """
    Adapter class that loads and manages all ML models.
    Provides a unified predict() interface that fuses rule-based and ML predictions.
    """
    
    def __init__(self):
        """Initialize the model adapter and load all available models."""
        self.tfidf = None
        self.distil = None
        self.rules = None
        self.fusion = None
        self._load_models()
    
    def _load_models(self) -> None:
        """
        Load all available models and modules.
        Gracefully handles missing models with fallbacks.
        """
        # Load TF-IDF pipeline
        tfidf_path = settings.TFIDF_MODEL_DIR
        try:
            from ml.tfidf_pipeline import TfidfPipeline
            p = TfidfPipeline()
            if os.path.exists(tfidf_path):
                p.load(tfidf_path)
                self.tfidf = p
                print(f"✓ TF-IDF model loaded from {tfidf_path}")
            else:
                print(f"⚠ TF-IDF model directory not found: {tfidf_path}")
        except Exception as e:
            self.tfidf = None
            print(f"⚠ TF-IDF load failed: {e}")
        
        # Load DistilBERT (optional)
        dist_path = settings.DISTILBERT_DIR
        try:
            from ml.distilbert_model import DistilBertWrapper
            if os.path.exists(dist_path):
                d = DistilBertWrapper(dist_path)
                d.load(dist_path)
                self.distil = d
                print(f"✓ DistilBERT model loaded from {dist_path}")
            else:
                print(f"ℹ DistilBERT model directory not found (optional): {dist_path}")
        except Exception as e:
            self.distil = None
            print(f"ℹ DistilBERT load failed (optional): {e}")
        
        # Load rules module
        try:
            import ml.rules as rules
            self.rules = rules
            print("✓ Rules module loaded")
        except Exception as e:
            self.rules = None
            print(f"⚠ Rules module load failed: {e}")
        
        # Load fusion module
        try:
            import ml.fusion as fusion
            self.fusion = fusion
            print("✓ Fusion module loaded")
        except Exception as e:
            self.fusion = None
            print(f"ℹ Fusion module not available (will use fallback): {e}")
    
    def predict(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Predict category for a transaction text.
        Combines rule-based and ML predictions using fusion logic.
        
        Args:
            text: Transaction description text
            meta: Optional metadata (MCC code, time, etc.)
            
        Returns:
            Dictionary with:
                - label: Predicted category
                - confidence: Confidence score (0-1)
                - rationale: Explanation with rule_hits and top_tokens
                - model_used: Which model(s) were used
                
        Raises:
            RuntimeError: If no models are available
        """
        if meta is None:
            meta = {}
        
        # 1) Apply rule-based classification
        rule_output = None
        if self.rules:
            try:
                rule_output = self.rules.apply_rules(text, meta)
                # Expected format: {'label': str, 'confidence': float, 'matches': list}
            except Exception as e:
                print(f"Rule application error: {e}")
                rule_output = None
        
        # 2) Apply ML model (prefer DistilBERT, fallback to TF-IDF)
        ml_output = None
        model_used = "none"
        
        if self.distil:
            try:
                ml_output = self.distil.predict([text])[0]
                model_used = "distilbert"
            except Exception as e:
                print(f"DistilBERT prediction error: {e}")
                ml_output = None
        
        if ml_output is None and self.tfidf:
            try:
                ml_output = self.tfidf.predict([text])[0]
                model_used = "tfidf"
            except Exception as e:
                print(f"TF-IDF prediction error: {e}")
                ml_output = None
        
        if ml_output is None and rule_output is None:
            raise RuntimeError("No models available for prediction")
        
        # 3) Fuse predictions
        if self.fusion and rule_output and ml_output:
            try:
                fused = self.fusion.fuse(rule_output, ml_output)
                fused["model_used"] = "fusion"
                return fused
            except Exception as e:
                print(f"Fusion error: {e}, falling back to simple merge")
        
        # Fallback: simple merge logic
        if rule_output and ml_output:
            # If rule has high confidence, prefer it
            if rule_output.get("confidence", 0) > 0.9:
                final_label = rule_output["label"]
                final_confidence = rule_output["confidence"]
            else:
                # Otherwise use ML
                final_label = ml_output.get("label")
                final_confidence = ml_output.get("confidence", 0.0)
            
            fused = {
                "label": final_label,
                "confidence": final_confidence,
                "rationale": {
                    "rule_hits": rule_output.get("matches", []),
                    "top_tokens": ml_output.get("top_tokens", [])
                },
                "model_used": f"{model_used}_with_rules"
            }
        elif rule_output:
            # Only rules available
            fused = {
                "label": rule_output["label"],
                "confidence": rule_output.get("confidence", 0.95),
                "rationale": {
                    "rule_hits": rule_output.get("matches", []),
                    "top_tokens": []
                },
                "model_used": "rules_only"
            }
        else:
            # Only ML available
            fused = {
                "label": ml_output.get("label"),
                "confidence": ml_output.get("confidence", 0.0),
                "rationale": {
                    "rule_hits": [],
                    "top_tokens": ml_output.get("top_tokens", [])
                },
                "model_used": model_used
            }
        
        return fused
    
    def get_model_status(self) -> Dict[str, bool]:
        """
        Get the status of all loaded models.
        
        Returns:
            Dictionary with model availability status
        """
        return {
            "tfidf": self.tfidf is not None,
            "distilbert": self.distil is not None,
            "rules": self.rules is not None,
            "fusion": self.fusion is not None
        }
