import re
import json
import pandas as pd
from typing import List

NORMALIZE_MAP_PATH = "ml/maps.json"

def _load_map():
    with open(NORMALIZE_MAP_PATH, "r", encoding="utf8") as f:
        return json.load(f)

NORMALIZE_MAP = _load_map()

VAGUE_WORDS = {
    "payment", "paid", "food", "misc", "transfer", "upi", "expense"
}

INTENT_KEYWORDS = {
    "meeting": "business",
    "client": "business",
    "office": "business",
    "hotel": "business",
    "flight": "business",
    "cab": "business",
    "uber": "business",
    "ola": "business",

    "dinner": "personal",
    "lunch": "personal",
    "movie": "personal",
    "shopping": "personal",
    "party": "personal"
}

def normalize_text(text: str) -> str:
    if text is None:
        return ""

    t = text.lower()
    t = re.sub(r"\d+", " ", t)          
    t = re.sub(r"[^a-z\s]", " ", t)     
    t = re.sub(r"\s+", " ", t).strip()

    for k in sorted(NORMALIZE_MAP.keys(), key=lambda x: -len(x)):
        if k in t:
            t = t.replace(k, NORMALIZE_MAP[k])

    return t

def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(normalize_text)


def detect_low_quality_note(text: str) -> bool:
    """
    Flags notes that are too vague to trust.
    Used for:
    - risk flags
    - confidence penalties
    - suggested user actions
    """
    if not text:
        return True

    t = text.strip().lower()
    words = t.split()

    # very short notes
    if len(words) <= 2:
        return True

    # vague single-word notes
    if t in VAGUE_WORDS:
        return True

    return False


def extract_intent_phrases(text: str) -> List[str]:
    """
    Extracts explicit intent signals from transaction notes.
    Used ONLY for explanations (not prediction).
    """
    if not text:
        return []

    t = text.lower()
    intents = []

    for key in INTENT_KEYWORDS:
        if key in t:
            intents.append(key)

    return intents
