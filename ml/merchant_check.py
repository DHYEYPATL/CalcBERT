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

def verify_merchant(name: str, note: str = None) -> dict:
    """
    Verify merchant by checking both merchant name and note text.
    This helps catch cases where merchant name is masked (e.g., "******7009")
    but the actual merchant is mentioned in the note (e.g., "uber_ride").
    """
    # Check merchant name first
    if name:
        key = normalize_text(name)
        merchant = REGISTRY.get(key)
        if merchant:
            return {
                "verified": bool(merchant.get("verified", False)),
                "merchant_type": merchant.get("type"),
                "warning": None
            }
    
    # If merchant name doesn't match, check note for merchant keywords
    if note:
        note_normalized = normalize_text(note)
        # Check if any registry key appears in the note
        for registry_key, merchant_info in REGISTRY.items():
            if registry_key in note_normalized and len(registry_key) > 2:  # Avoid matching very short keys
                return {
                    "verified": bool(merchant_info.get("verified", False)),
                    "merchant_type": merchant_info.get("type"),
                    "warning": None,
                    "matched_from": "note"
                }
    
    # No match found
    return {
        "verified": False,
        "merchant_type": None,
        "warning": WARNING_MSG
    }
