import statistics

def extract_behavioral_features(user_data):
    """
    Analyzes historical transactions mathematically to synthesize clustering variables.
    Returns: [risk_tolerance, spending_volatility, impulse_index, merchant_diversity, weekend_spike]
    """
    txs = user_data.get("transactions", [])
    if not txs:
         return [0.5, 0.1, 0.1, 0.5, 1.0] # Flat median defaults if no data exists
         
    avg_tx = statistics.mean(txs)
    
    # Measure transaction standard deviations
    volatility = statistics.stdev(txs) / max(avg_tx, 1.0) if len(txs) > 1 else 0.0
    
    # Determine impulse spikes: frequency of high anomalous transactions pulling on sums
    max_tx = max(txs)
    impulse = min((max_tx - avg_tx) / max(sum(txs), 1.0) * 2, 1.0)
    
    # Risk factor corresponds negatively to chaotic variance behaviors
    risk = max(0.1, 1.0 - volatility * 0.5)
    
    # Proxied diversity ratio tracking unique transactions
    diversity = min(len(set(txs)) / max(len(txs), 1), 1.0)
    
    # Weekend spike probability emulation based loosely upon variance anomalies
    weekend_spike = 1.0 + (volatility * 0.4)
    
    return [risk, volatility, impulse, diversity, weekend_spike]
