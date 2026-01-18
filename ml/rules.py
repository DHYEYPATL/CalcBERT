

import re
from typing import Dict, Any, List, Optional


# Rule patterns for different categories
RULES = {
    "Coffee & Beverages": [
        r"\bstarbucks?\b",
        r"\bstarbcks?\b",
        r"\bcoffee\b",
        r"\bcafe\b",
        r"\bdunkin\b",
        r"\bpeets?\b",
    ],
    "Transportation": [
        r"\buber\b",
        r"\blyft\b",
        r"\btaxi\b",
        r"\bcab\b",
        r"\bmetro\b",
        r"\bsubway\b",
        r"\btrain\b",
        r"\bbus\b",
    ],
    "Restaurant & Dining": [
        r"\bmcdonalds?\b",
        r"\bmcdonald\b",
        r"\bburger\b",
        r"\bpizza\b",
        r"\brestaurant\b",
        r"\bdining\b",
        r"\bkfc\b",
        r"\bsubway\b",
    ],
    "Online Shopping": [
        r"\bamazon\b",
        r"\bebay\b",
        r"\bshopify\b",
        r"\betsy\b",
    ],
    "Groceries": [
        r"\bwalmart\b",
        r"\btarget\b",
        r"\bcostco\b",
        r"\bwhole\s*foods?\b",
        r"\bgrocery\b",
        r"\bsupermarket\b",
    ],
    "Entertainment": [
        r"\bnetflix\b",
        r"\bspotify\b",
        r"\bhulu\b",
        r"\bdisney\b",
        r"\bcinema\b",
        r"\bmovie\b",
        r"\btheater\b",
    ],
    "Gas & Fuel": [
        r"\bshell\b",
        r"\bchevron\b",
        r"\bexxon\b",
        r"\bmobil\b",
        r"\bgas\s*station\b",
        r"\bfuel\b",
    ],
    "Healthcare": [
        r"\bcvs\b",
        r"\bwalgreens\b",
        r"\bpharmacy\b",
        r"\bdoctor\b",
        r"\bhospital\b",
        r"\bmedical\b",
    ],
}


def apply_rules(text: str, meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
   
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Also try without word boundaries for better matching with special chars
    # e.g., "#starbucks#" or "626starbucks765" should still match
    text_for_fuzzy = re.sub(r'[^a-z0-9\s]', ' ', text_lower)  # Replace special chars with spaces
    text_for_fuzzy = ' ' + text_for_fuzzy + ' '  # Add spaces for boundary matching
    
    # Check each category's rules
    for category, patterns in RULES.items():
        matches = []
        for pattern in patterns:
            # Try strict word boundary match first
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(pattern.replace(r"\b", "").replace("?", ""))
            else:
                # Fallback: extract keyword and check if it exists (handles special chars)
                keyword = pattern.replace(r"\b", "").replace("?", "").replace("\\", "")
                if keyword in text_lower:
                    matches.append(keyword)
        
        if matches:
            # High confidence if rule matched
            return {
                "label": category,
                "confidence": 0.95,
                "matches": matches,
                "source": "rule-based"
            }
    
    # No rule matched
    return None


def get_all_categories() -> List[str]:
    
    return list(RULES.keys())


def add_rule(category: str, pattern: str) -> None:
    
    if category not in RULES:
        RULES[category] = []
    RULES[category].append(pattern)
