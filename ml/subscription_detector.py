import pandas as pd
from ml.data_pipeline import normalize_text

def detect_subscription(
    history_df: pd.DataFrame,
    merchant: str,
    amount: float,
    amount_tolerance: float = 0.05,
    min_occurrences: int = 3
) -> dict:
   

    # Default response (defensive)
    result = {
        "is_subscription": False,
        "period": None,
        "count": 0,
        "reason": "No history"
    }

    if history_df is None or history_df.empty:
        return result

    df = history_df.copy()
    df["merchant_norm"] = df["merchant"].apply(normalize_text)
    merchant_key = normalize_text(merchant)
    df = df[df["merchant_norm"] == merchant_key]

    result["count"] = len(df)

    if len(df) < min_occurrences:
        result["reason"] = "Insufficient occurrences"
        return result

    df = df[
        (df["amount"] >= amount * (1 - amount_tolerance)) &
        (df["amount"] <= amount * (1 + amount_tolerance))
    ]

    result["count"] = len(df)

    if len(df) < min_occurrences:
        result["reason"] = "Amount variance too high"
        return result

    dates = pd.to_datetime(df["date"]).sort_values()
    gaps = dates.diff().dt.days.dropna()

    if len(gaps) < min_occurrences - 1:
        result["reason"] = "Not enough intervals"
        return result

    monthly_hits = gaps.between(27, 32).sum()
    weekly_hits = gaps.between(6, 8).sum()

    if monthly_hits >= 2:
        result.update({
            "is_subscription": True,
            "period": "monthly",
            "reason": "Repeated monthly charge pattern"
        })
        return result

    if weekly_hits >= 2:
        result.update({
            "is_subscription": True,
            "period": "weekly",
            "reason": "Repeated weekly charge pattern"
        })
        return result

    result["reason"] = "Irregular payment intervals"
    return result
