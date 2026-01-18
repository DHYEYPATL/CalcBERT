def fuse(rule_output, ml_output, tfidf_output):
    RULE_HIGH = 0.9
    TFIDF_HIGH = 0.85

    # 1️⃣ Rule override
    if rule_output and rule_output.get("confidence", 0) >= RULE_HIGH:
        return {
            "decision": "allow",
            "final_category": rule_output.get("label", "Unknown"),
            "final_confidence": rule_output.get("confidence", 0.0),
            "model_used": "rules",
            "rationale": {
                "rule_hits": rule_output.get("matches", [])
            }
        }

    # 2️⃣ TF-IDF override
    if tfidf_output and tfidf_output.get("confidence", 0) >= TFIDF_HIGH:
        return {
            "decision": "warn",
            "final_category": tfidf_output.get("label", "Unknown"),
            "final_confidence": tfidf_output.get("confidence", 0.0),
            "model_used": "tfidf",
            "rationale": {
                "risk_flags": tfidf_output.get("risk_flags", {})
            }
        }

    # 3️⃣ DistilBERT fallback
    if ml_output:
        return {
            "decision": "warn" if ml_output.get("confidence", 0) < 0.8 else "allow",
            "final_category": ml_output.get("label", "Unknown"),
            "final_confidence": ml_output.get("confidence", 0.0),
            "model_used": "distilbert",
            "rationale": {
                "top_tokens": ml_output.get("top_tokens", [])
            }
        }

    return {
        "decision": "block",
        "final_category": "unknown",
        "final_confidence": 0.0,
        "model_used": "none",
        "rationale": {}
    }
