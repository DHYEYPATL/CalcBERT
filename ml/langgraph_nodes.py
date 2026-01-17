
from ml.distilbert_model import DistilBertWrapper
from ml.fusion import fuse

# Load model once for reuse
distilbert = DistilBertWrapper("saved_models/distilbert")

def distilbert_node(state: dict) -> dict:
  
    text = state.get("text", "")
    state["bert_output"] = distilbert.predict([text])[0]  
    return state

def fusion_node(state: dict) -> dict:
    
    state["final_decision"] = fuse(
        rule_output=state.get("rule_output"),
        ml_output=state.get("bert_output"),
        tfidf_output=state.get("tfidf_output")
    )
    return state

# Example additional nodes:
def explanation_node(state: dict) -> dict:
    
    
    state["explanation"] = state["final_decision"].get("rationale", {})
    return state
