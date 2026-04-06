import numpy as np

def get_persona(data=None, mode=None):
    if mode == "mock":
        return {
            "persona": "Balanced User",
            "confidence": 0.7,
            "behavior_flags": []
        }

    data = data if isinstance(data, dict) else {}
    tx = data.get("transactions", [])

    if not tx:
        return {}

    mean = np.mean(tx)
    std = np.std(tx)
    spike = max(tx) / (mean + 1)

    if std < 50:
        persona = "Conservative Saver"
        flags = []
    elif spike > 5:
        persona = "Impulsive Spender"
        flags = ["High impulse spending"]
    else:
        persona = "Balanced User"
        flags = []

    return {
        "persona": persona,
        "confidence": 0.8,
        "behavior_flags": flags
    }
