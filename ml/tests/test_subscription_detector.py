import pandas as pd
from ml.subscription_detector import detect_subscription

def test_monthly_subscription_detected():
    data = {
        "merchant": ["netflix", "netflix", "netflix"],
        "amount": [499, 499, 499],
        "date": ["2024-01-01", "2024-02-01", "2024-03-01"]
    }
    df = pd.DataFrame(data)

    result = detect_subscription(df, "netflix", 499)

    assert result["is_subscription"] is True
    assert result["period"] == "monthly"
    assert result["count"] == 3

def test_not_subscription_insufficient_data():
    df = pd.DataFrame({
        "merchant": ["uber"],
        "amount": [200],
        "date": ["2024-01-01"]
    })

    result = detect_subscription(df, "uber", 200)
    assert result["is_subscription"] is False
