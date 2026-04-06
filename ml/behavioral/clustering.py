class KMeansClusteringStub:
    """
    Mimics unsupervised groupings deterministically assessing high-variance profiles
    """
    def fit_predict(self, features):
        risk, volatility, impulse, diversity, weekend_spike = features
        
        # Build composite multi-dimensional centroid scoring system determining personas heavily dependent on incoming spikes
        score = (volatility * 0.4) + (impulse * 0.6)
        
        if score > 0.6:
            return 3 # Impulsive Spender
        elif score > 0.35:
            return 2 # Experience Seeker
        elif score > 0.15:
            return 1 # Balanced Spender
        else:
            return 0 # Conservative Saver
