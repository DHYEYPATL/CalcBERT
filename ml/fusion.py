def fuse(rule_output, ml_output, tfidf_output, weights=None):
    
    weights = weights or {"rule": 0.6, "ml": 0.3, "tfidf": 0.1}
    RULE_HIGH_CONF = 0.9
    TFIDF_HIGH_CONF = 0.10  
    
    if rule_output and rule_output.get("confidence", 0) >= RULE_HIGH_CONF:
        final = rule_output.copy()
        final["model_used"] = "rule"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []),
            "top_tokens": ml_output.get("top_tokens", []) if ml_output else [],
            "weighting": weights,
            "notes": "Rule wins with high confidence."
        }
        return final

    
    if tfidf_output and tfidf_output.get("confidence", 0) >= TFIDF_HIGH_CONF:
        final = tfidf_output.copy()
        final["model_used"] = "tfidf"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []) if rule_output else [],
            "top_tokens": tfidf_output.get("top_tokens", []),
            "weighting": weights,
            "notes": "TF-IDF override (confidence due to recent feedback)."
        }
        return final

    
    if ml_output:
        final = ml_output.copy()
        final["model_used"] = "distilbert"
        final["rationale"] = {
            "rule_hits": rule_output.get("matches", []) if rule_output else [],
            "top_tokens": ml_output.get("top_tokens", []),
            "weighting": weights,
            "notes": "DistilBERT used: no strong rule or TF-IDF match."
        }
        return final

    return rule_output or {"label": "Unknown", "confidence": 0.0, "rationale": {}, "model_used": "none"}
