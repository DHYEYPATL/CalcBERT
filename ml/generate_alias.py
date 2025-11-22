import pandas as pd
import json
import re
from collections import defaultdict
from fuzzywuzzy import fuzz

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def cluster_merchants(merchants):
    clusters = []
    used = set()

    for i in range(len(merchants)):
        if i in used:
            continue
        
        base = merchants[i]
        cluster = [base]

        for j in range(len(merchants)):
            if j in used or j == i:
                continue

            
            if fuzz.token_sort_ratio(base, merchants[j]) >= 80:
                cluster.append(merchants[j])
                used.add(j)

        used.add(i)
        clusters.append(cluster)

    return clusters

def pick_canonical(cluster):
    
    clean_candidates = [c for c in cluster if not re.search(r"\d", c)]
    if clean_candidates:
        cluster = clean_candidates

    
    return sorted(cluster, key=lambda x: (-len(x), x))[0]

def generate_map(csv_path="data/GHCI_clean.csv"):
    df = pd.read_csv(csv_path)

    merchants_raw = df["transaction_text"].fillna("").astype(str).tolist()
    cleaned = [clean_text(t) for t in merchants_raw]

  
    merchants_unique = list(set(cleaned))

    clusters = cluster_merchants(merchants_unique)

    normalize_map = {}

    for cluster in clusters:
        canonical = pick_canonical(cluster)
        for alias in cluster:
            if alias != canonical:
                normalize_map[alias] = canonical

    with open("ml/maps.json", "w") as f:
        json.dump(normalize_map, f, indent=2)

    print("Generated maps.json with", len(normalize_map), "entries.")

if __name__ == "__main__":
    generate_map("data/GHCI_clean.csv")
