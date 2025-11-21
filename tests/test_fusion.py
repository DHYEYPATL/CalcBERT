from ml.fusion import fuse

def test_high_conf_rule():
    rule = {"label": "Coffee", "confidence": 0.95, "matches": ['starbucks']}
    ml = {"label": "Fuel", "confidence": 0.8, "top_tokens": [{"token": "petrol", "score": 0.5}]}
    result = fuse(rule, ml)
    assert result['label'] == "Coffee"
    assert result['model_used'] == "rule"
