from ml.distilbert_model import DistilBertWrapper
from ml.fusion import fuse

# Lazy initialization with error handling
_distilbert_instance = None

def _get_distilbert():
    global _distilbert_instance
    if _distilbert_instance is None:
        try:
            _distilbert_instance = DistilBertWrapper("saved_models/distilbert")
        except Exception as e:
            print(f"⚠ DistilBERT initialization failed: {e}")
            return None
    return _distilbert_instance


def distilbert_node(state: dict) -> dict:
    text = state.get("text", "")
    distilbert = _get_distilbert()
    if distilbert:
        try:
            predictions = distilbert.predict([text])
            if predictions and len(predictions) > 0:
                state["bert_output"] = predictions[0]
            else:
                state["bert_output"] = None
        except Exception as e:
            print(f"⚠ DistilBERT prediction error: {e}")
            state["bert_output"] = None
    else:
        state["bert_output"] = None
    return state


def fusion_node(state: dict) -> dict:
    state["final_decision"] = fuse(
        rule_output=state.get("rule_output"),
        ml_output=state.get("bert_output"),
        tfidf_output=state.get("tfidf_output")
    )
    return state


def explanation_node(state: dict) -> dict:
    state["explanation"] = state["final_decision"].get("rationale", {})
    return state
