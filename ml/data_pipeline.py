import re
import json
import pandas as pd

NORMALIZE_MAP_PATH = "ml/maps.json"

def _load_map():
    with open(NORMALIZE_MAP_PATH, "r", encoding="utf8") as f:
        return json.load(f)

NORMALIZE_MAP = _load_map()

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    for k in sorted(NORMALIZE_MAP.keys(), key=lambda x: -len(x)):
        if k in t:
            t = t.replace(k, NORMALIZE_MAP[k])

    return t

def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(normalize_text)
