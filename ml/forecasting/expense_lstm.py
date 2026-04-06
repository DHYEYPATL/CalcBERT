import statistics

class LSTMExpenseStub:
    """
    Simulates an LSTM model predicting the long-term compounding momentum trend of spending.
    """
    def predict(self, user_data):
        spends = user_data.get("monthly_spend", []) if isinstance(user_data, dict) else []
        
        if len(spends) > 1:
            avg_spend = statistics.mean(spends)
            trend = spends[-1] / max(spends[0], 1.0)
        elif len(spends) == 1:
            avg_spend = spends[0]
            trend = 1.0
        else:
            avg_spend = 45000
            trend = 1.02
            
        trend = min(max(trend, 0.8), 1.25)
        
        predictions = {}
        current = avg_spend
        for month in ["2024-05", "2024-06", "2024-07", "2024-08"]:
            current *= trend
            predictions[month] = current
            
        return predictions, trend
