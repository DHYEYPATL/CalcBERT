class XGBoostIncomeRegressorStub:
    """
    A deterministic heuristic model acting as a surrogate for XGBoost.
    It builds complex mathematical relationships utilizing extracted features
    to output highly dynamic and realistic ranges instead of static returns.
    """
    def __init__(self):
        self.is_trained = True

    def predict(self, features):
        base = features["base_income_estimate"]
        trend_impact = (features["monthly_income_trend"] - 1.0) * 0.4
        spike_penalty = features["freelance_spike_index"] * 0.1
        bonus_boost = features["bonus_frequency"] * 0.03
        cagr_boost = features["investment_cagr"] * 0.2
        seasonality = features["seasonality_factor"]
        
        # Calculate core deterministic multiplier responding distinctly to feature interactions
        multiplier = 1.0 + trend_impact - spike_penalty + bonus_boost + cagr_boost
        predicted = base * multiplier * seasonality
        
        # Confidence interval expands if empirical variance/spikes are high (mathematical uncertainty)
        margin = 0.1 + (features["freelance_spike_index"] * 0.2)
        low_bound = predicted * (1 - margin)
        high_bound = predicted * (1 + margin)
        
        return {
            "predicted": round(max(predicted, 10000), 2),
            "range": [round(max(low_bound, 9000), 2), round(high_bound, 2)]
        }
