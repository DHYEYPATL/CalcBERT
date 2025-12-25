import pandas as pd
from ml.summary_builder import build_daily_summary

def test_summary_builder_basic():
    df = pd.DataFrame({
        "merchant": ["Swiggy", "Uber", "Swiggy"],
        "category": ["Food", "Transport", "Food"],
        "confidence": [0.8, 0.9, 0.7],
        "amount": [300, 200, 400],
        "corrected": [0, 1, 0]
    })

    summary = build_daily_summary(df)

    assert summary["today_spend_by_category"]["Food"] == 700
    assert summary["manual_corrections"] == 1
    assert "Swiggy" in summary["top_merchants"]
