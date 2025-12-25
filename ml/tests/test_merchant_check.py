from ml.merchant_check import verify_merchant

def test_known_merchant_verified():
    result = verify_merchant("swiggy")
    assert result["verified"] is True
    assert result["merchant_type"] == "food_delivery"
    assert result["warning"] is None

def test_unknown_merchant_flagged():
    result = verify_merchant("someRandomUPI123")
    assert result["verified"] is False
    assert result["merchant_type"] is None
    assert "NEVER IDENTIFIED" in result["warning"]

def test_empty_merchant():
    result = verify_merchant("")
    assert result["verified"] is False
