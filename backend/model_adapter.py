import os
import sys
from typing import Dict, Any, Optional


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.config import settings


class ModelAdapter:
   
    
    def __init__(self):
        
        self.tfidf = None
        self.distil = None
        self.rules = None
        self.fusion = None
        self._load_models()
    
    def _load_models(self) -> None:
        
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
        
        
        try:
            import ml.rules as rules
            self.rules = rules
            print("✓ Rules module loaded")
        except Exception as e:
            self.rules = None
            print(f"⚠ Rules module load failed: {e}")
        
        
        try:
            import ml.fusion as fusion
            self.fusion = fusion
            print("✓ Fusion module loaded")
        except Exception as e:
            self.fusion = None
            print(f"ℹ Fusion module not available (will use fallback): {e}")
    
    def predict(self, text: str, meta: Optional[Dict] = None) -> Dict[str, Any]:
        
        if meta is None:
            meta = {}
        
       
        rule_output = None
        if self.rules:
            try:
                rule_output = self.rules.apply_rules(text, meta)
                
            except Exception as e:
                print(f"Rule application error: {e}")
                rule_output = None
        
        
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

        tfidf_output = None
        if self.tfidf:
            try:
                tfidf_output = self.tfidf.predict([text])[0]
            except Exception as e:
                print(f"TF-IDF prediction error: {e}")
                tfidf_output = None

        if self.fusion and (rule_output or ml_output or tfidf_output):
            try:
                fused = self.fusion.fuse(rule_output, ml_output, tfidf_output)
                return fused
            except Exception as e:
                print(f"Fusion error: {e}, falling back to simple merge")
       

        if rule_output and ml_output:
            
            if rule_output.get("confidence", 0) > 0.9:
                final_label = rule_output["label"]
                final_confidence = rule_output["confidence"]
            else:
                
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
    
    def reload_tfidf_model(self) -> bool:
        
        tfidf_path = settings.TFIDF_MODEL_DIR
        try:
            from ml.tfidf_pipeline import TfidfPipeline
            p = TfidfPipeline()
            if os.path.exists(tfidf_path):
                p.load(tfidf_path)
                self.tfidf = p
                print(f"✓ TF-IDF model reloaded from {tfidf_path}")
                return True
            else:
                print(f"⚠ TF-IDF model directory not found: {tfidf_path}")
                return False
        except Exception as e:
            print(f"⚠ TF-IDF reload failed: {e}")
            return False
    
    def get_model_status(self) -> Dict[str, bool]:
        
        return {
            "tfidf": self.tfidf is not None,
            "distilbert": self.distil is not None,
            "rules": self.rules is not None,
            "fusion": self.fusion is not None
        }
    
    # ========================================================================
    # v2 Backend: Extended Methods
    # ========================================================================
    
    def predict_structured(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict with structured output for workflow integration.
        Returns separate outputs from each model component.
        
        Args:
            state: State dictionary with 'note' and optional 'meta'
            
        Returns:
            Dictionary with:
            {
                "rule_output": {...} or None,
                "tfidf_output": {...} or None,
                "bert_output": {...} or None
            }
        """
        text = state.get("note", "")
        meta = state.get("meta", {})
        
        result = {
            "rule_output": None,
            "tfidf_output": None,
            "bert_output": None
        }
        
        # Get rule output
        if self.rules:
            try:
                result["rule_output"] = self.rules.apply_rules(text, meta)
            except Exception as e:
                print(f"Rule prediction error: {e}")
        
        # Get TF-IDF output
        if self.tfidf:
            try:
                result["tfidf_output"] = self.tfidf.predict([text])[0]
            except Exception as e:
                print(f"TF-IDF prediction error: {e}")
        
        # Get DistilBERT output (optional)
        if self.distil:
            try:
                result["bert_output"] = self.distil.predict([text])[0]
            except Exception as e:
                print(f"DistilBERT prediction error: {e}")
        
        return result
    
    def detect_low_quality_note(self, text: str) -> Dict[str, Any]:
        """
        Detect if transaction note is low quality.
        
        Args:
            text: Transaction note text
            
        Returns:
            {
                "is_low_quality": bool,
                "reason": str,
                "suggestions": list[str]
            }
        """
        if not text or len(text.strip()) == 0:
            return {
                "is_low_quality": True,
                "reason": "Empty note",
                "suggestions": ["Add a description of the expense"]
            }
        
        text = text.strip()
        words = text.split()
        
        # Check for very short notes
        if len(words) < 3:
            return {
                "is_low_quality": True,
                "reason": "Note too short",
                "suggestions": [
                    "Add more context about the expense",
                    "Include purpose and business justification"
                ]
            }
        
        # Check for generic/vague terms
        vague_terms = ["stuff", "things", "misc", "miscellaneous", "expense", "payment"]
        if any(term in text.lower() for term in vague_terms):
            return {
                "is_low_quality": True,
                "reason": "Vague description",
                "suggestions": [
                    "Be more specific about what was purchased",
                    "Include business purpose"
                ]
            }
        
        # Good quality note
        return {
            "is_low_quality": False,
            "reason": "Adequate detail",
            "suggestions": []
        }
    
    def detect_merchant_type(self, merchant: str) -> str:
        """
        Detect merchant verification status.
        
        Args:
            merchant: Merchant name
            
        Returns:
            'verified', 'known', or 'unknown'
        """
        # List of verified/known merchants (in production, this would be a database)
        verified_merchants = [
            "swiggy", "zomato", "uber", "ola", "amazon", "flipkart",
            "starbucks", "mcdonald", "dominos", "pizza hut",
            "reliance", "big bazaar", "dmart", "more",
            "irctc", "makemytrip", "goibibo", "oyo"
        ]
        
        merchant_lower = merchant.lower()
        
        # Check if verified
        for verified in verified_merchants:
            if verified in merchant_lower:
                return "verified"
        
        # Check if it looks like a known pattern (has @upi, etc.)
        if "@" in merchant_lower or "upi" in merchant_lower:
            return "known"
        
        # Unknown merchant
        return "unknown"
    
    def detect_subscription(
        self,
        merchant: str,
        amount: float,
        note: str
    ) -> Dict[str, Any]:
        """
        Detect if transaction is likely a subscription.
        
        Args:
            merchant: Merchant name
            amount: Transaction amount
            note: Transaction note
            
        Returns:
            {
                "is_subscription": bool,
                "confidence": float,
                "pattern": str
            }
        """
        # Subscription keywords
        subscription_keywords = [
            "subscription", "monthly", "annual", "yearly", "recurring",
            "membership", "premium", "pro", "plan"
        ]
        
        # Known subscription merchants
        subscription_merchants = [
            "netflix", "spotify", "amazon prime", "disney", "hotstar",
            "youtube premium", "apple music", "office 365", "adobe",
            "dropbox", "google one", "icloud"
        ]
        
        merchant_lower = merchant.lower()
        note_lower = note.lower()
        
        # Check for subscription merchants
        for sub_merchant in subscription_merchants:
            if sub_merchant in merchant_lower:
                return {
                    "is_subscription": True,
                    "confidence": 0.95,
                    "pattern": "known_subscription_merchant"
                }
        
        # Check for subscription keywords in note
        keyword_matches = sum(1 for kw in subscription_keywords if kw in note_lower)
        if keyword_matches > 0:
            return {
                "is_subscription": True,
                "confidence": min(0.7 + (keyword_matches * 0.1), 0.95),
                "pattern": "subscription_keywords"
            }
        
        # Check for round amounts (common in subscriptions)
        if amount in [99, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1499, 1999]:
            return {
                "is_subscription": True,
                "confidence": 0.6,
                "pattern": "common_subscription_amount"
            }
        
        # Not a subscription
        return {
            "is_subscription": False,
            "confidence": 0.0,
            "pattern": "none"
        }

