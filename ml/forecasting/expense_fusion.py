import numpy as np

def forecast_expense(data=None, mode=None):
    if mode == "mock":
        return {
            "monthly_spend_forecast": {"m1": 1000},
            "subscription_growth_rate": 0.1,
            "cash_flow_deficit_months": [],
            "confidence": 0.8
        }

    data = data if isinstance(data, dict) else {}
    spends = data.get("monthly_spend", [])
    income = data.get("projected_monthly_income", 0)

    if not spends:
        return {}

    trend = spends[-1] - spends[0]
    avg = np.mean(spends)

    # simulate 3 models
    prophet = avg + trend * 0.5
    lstm = avg + trend * 1.0
    arima = spends[-1] + trend * 0.3

    final = np.median([prophet, lstm, arima])

    forecast = {
        "month1": final,
        "month2": final * 1.05,
        "month3": final * 1.1
    }

    deficit = [m for m, v in forecast.items() if v > income]

    return {
        "monthly_spend_forecast": forecast,
        "subscription_growth_rate": (spends[-1] - spends[0]) / (spends[0] + 1) if spends else 0,
        "cash_flow_deficit_months": deficit,
        "confidence": 0.8
    }
