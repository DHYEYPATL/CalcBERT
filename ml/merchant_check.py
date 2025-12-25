import json
import os
from ml.data_pipeline import normalize_text

# absolute path relative to THIS file
BASE_DIR = os.path.dirname(__file__)
REGISTRY_PATH = os.path.join(BASE_DIR, "merchant_registry.json")

def _load_registry():
    if not os.path.exists(REGISTRY_PATH):
        # fallback for tests / empty registry
        return {}
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

REGISTRY = _load_registry()

WARNING_MSG = "NEVER IDENTIFIED THIS MERCHANT; BE AWARE"

def verify_merchant(name: str) -> dict:
    if not name:
        return {
            "verified": False,
            "merchant_type": None,
            "warning": WARNING_MSG
        }

    key = normalize_text(name)
    merchant = REGISTRY.get(key)

    if not merchant:
        return {
            "verified": False,
            "merchant_type": None,
            "warning": WARNING_MSG
        }

    return {
        "verified": bool(merchant.get("verified", False)),
        "merchant_type": merchant.get("type"),
        "warning": None
    }
