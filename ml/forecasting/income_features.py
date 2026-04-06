import statistics

def extract_all_features(user_data):
    """
    Extracts deterministic features from raw input data so predictive models can
    react realistically to different user profiles and inputs dynamically.
    """
    features = {}
    incomes = user_data.get("monthly_incomes", [])
    
    if isinstance(incomes, list) and len(incomes) > 1:
        # Calculate real trend and variance to provide signals
        trend = incomes[-1] / max(incomes[0], 1.0)
        spike = statistics.stdev(incomes) / max(statistics.mean(incomes), 1.0) if len(incomes) > 2 else 0.1
    else:
        # Flat safe defaults
        trend = 1.05
        spike = 0.05
        
    features["monthly_income_trend"] = min(max(trend, 0.5), 2.0)
    features["freelance_spike_index"] = min(spike, 1.0)
    features["bonus_frequency"] = user_data.get("bonus_count", 1)
    features["investment_cagr"] = user_data.get("investment_cagr", 0.08)
    features["seasonality_factor"] = user_data.get("seasonality_factor", 1.0)
    
    # Estimate base income
    if incomes:
        features["base_income_estimate"] = statistics.mean(incomes) * 12
    else:
        features["base_income_estimate"] = user_data.get("base_income", 1000000)
        
    return features
