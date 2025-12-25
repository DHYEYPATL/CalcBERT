import pandas as pd

def build_daily_summary(df: pd.DataFrame, window_days: int = 1) -> dict:
    return {
        "today_spend_by_category": (
            df.groupby("category")["amount"].sum().to_dict()
            if not df.empty and "category" in df.columns
            else {}
        ),
        "average_confidence": (
            round(float(df["confidence"].mean()), 3)
            if "confidence" in df.columns and df["confidence"].notna().any()
            else None
        ),
        "manual_corrections": (
            int(df["corrected"].astype(int).sum())
            if "corrected" in df.columns
            else 0
        ),
        "top_merchants": (
            df["merchant"].value_counts().head(3).index.tolist()
            if "merchant" in df.columns
            else []
        )
    }
