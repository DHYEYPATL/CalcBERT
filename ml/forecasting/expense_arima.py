class ARIMAExpenseStub:
    """
    Simulates an ARIMA model focusing on auto-regressive short term bounce-backs or mean reversions.
    """
    def predict(self, user_data, base_spend):
        spends = user_data.get("monthly_spend", []) if isinstance(user_data, dict) else []
        
        # If last month dropped significantly, assume a bounce back
        if len(spends) > 1 and spends[-1] < spends[-2] * 0.9:
            correction = 1.05
        # If last month spiked, assume mean reversion dropping it
        elif len(spends) > 1 and spends[-1] > spends[-2] * 1.1:
            correction = 0.95
        else:
            correction = 1.0
            
        predictions = {}
        current = base_spend
        for month in ["2024-05", "2024-06", "2024-07", "2024-08"]:
            current *= correction
            predictions[month] = current
            # Fade the correction towards 1.0 over time
            correction = 1.0 + (correction - 1.0) * 0.5
            
        return predictions
