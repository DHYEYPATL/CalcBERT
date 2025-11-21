def fuse(rule_output, ml_output, weights=None):
    weights = weights or {"rule": 0.6, "ml": 0.4}
    RULE_HIGH_CONF = 0.9
    if rule_output and rule_output.get("confidence", 0) >= RULE_HIGH_CONF:
        final = rule_output.copy()
        final["model_used"] = "rule"
    else:
        final = ml_output.copy()
        final["model_used"] = "ml"
    final["rationale"] = {
        "rule_hits": rule_output.get("matches", []) if rule_output else [],
        "top_tokens": ml_output.get("top_tokens", []) if ml_output else [],
        "weighting": weights,
        "notes": "rule wins if high-confidence, else ML" if rule_output and rule_output.get("confidence", 0) >= RULE_HIGH_CONF else "ML used"
    }
    return final
