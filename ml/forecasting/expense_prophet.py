import statistics

class ProphetExpenseStub:
    """
    Simulates a Prophet model extracting cyclical seasonality and holiday factors.
    """
    def predict(self, user_data, base_spend):
        predictions = {}
        # Simple cyclic oscillation simulating month-over-month seasonality changes
        for i, month in enumerate(["2024-05", "2024-06", "2024-07", "2024-08"]):
            seasonality = 1.0 + 0.05 * (i % 2) - 0.02 * (i % 3)
            predictions[month] = base_spend * seasonality
        return predictions
