"""
Seed Data Generator for CalcBERT v2 Backend
Generates 3-month historical data for testing frontend analytics.
"""

import sqlite3
import json
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Use path relative to project root so it works from cwd or scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DB_PATH = os.path.join(_PROJECT_ROOT, "backend", "backend_feedback.db")

# =============================================================================
# Data Constants
# =============================================================================

MERCHANTS = {
    "Food": ["Swiggy", "Zomato", "Dominos", "Pizza Hut", "McDonalds", "Starbucks", "Cafe Coffee Day", "Haldirams"],
    "Transport": ["Uber", "Ola", "Rapido", "Metro Recharge", "Petrol Pump HP", "Indian Oil"],
    "Shopping": ["Amazon", "Flipkart", "Myntra", "Ajio", "DMart", "Big Bazaar", "Reliance Digital"],
    "Groceries": ["BigBasket", "Blinkit", "Zepto", "JioMart", "Nature's Basket"],
    "Entertainment": ["PVR Cinemas", "BookMyShow", "Netflix", "Spotify", "Gaming Zone"],
    "Utilities": ["Electricity Board", "Jio Fiber", "Airtel", "BSNL", "Gas Agency", "Water Board"],
    "Business": ["WeWork", "Hotel Taj", "Hotel Marriott", "Conference Center", "Airport Lounge"]
}

NOTES_TEMPLATES = {
    "Food": ["lunch with team", "client dinner", "office snacks", "team celebration", "working lunch", "quick bite"],
    "Transport": ["office commute", "client visit", "airport pickup", "site visit", "late night cab"],
    "Shopping": ["office supplies", "equipment purchase", "team gifts", "event decorations"],
    "Groceries": ["pantry supplies", "office refreshments", "meeting snacks"],
    "Entertainment": ["team outing", "client entertainment", "training session break"],
    "Utilities": ["monthly bill", "recharge", "quarterly payment"],
    "Business": ["client meeting", "conference", "business travel", "accommodation"]
}

UPI_SUFFIXES = ["@upi", "@paytm", "@ybl", "@okhdfcbank", "@okaxis", "@oksbi"]

# user-123 matches DEFAULT_USER_ID in frontend (src/constants/user.ts)
USER_IDS = ["user-123", "user_001", "user_002", "user_003", "emp_101", "emp_102", "mgr_001"]
USER_ROLES = ["employee", "employee", "employee", "manager", "admin"]

SUBSCRIPTIONS = [
    {"name": "Netflix", "amount": 649.0, "period": "monthly"},
    {"name": "Spotify Premium", "amount": 119.0, "period": "monthly"},
    {"name": "YouTube Premium", "amount": 129.0, "period": "monthly"},
    {"name": "Electricity Bill", "amount": 2500.0, "period": "monthly"},
    {"name": "Jio Fiber WiFi", "amount": 999.0, "period": "monthly"},
    {"name": "Airtel Mobile", "amount": 599.0, "period": "monthly"},
    {"name": "Amazon Prime", "amount": 1499.0, "period": "yearly"},
    {"name": "Gym Membership", "amount": 2000.0, "period": "monthly"},
]

# =============================================================================
# Helper Functions
# =============================================================================

def get_dates_for_3_months() -> List[datetime]:
    """Generate a list of dates spanning the last 3 months."""
    dates = []
    base = datetime.now()
    for days_ago in range(90):
        dates.append(base - timedelta(days=days_ago))
    return dates

def random_timestamp_for_date(date: datetime) -> int:
    """Generate a random timestamp for a given date."""
    hour = random.randint(8, 22)
    minute = random.randint(0, 59)
    dt = date.replace(hour=hour, minute=minute, second=random.randint(0, 59))
    return int(dt.timestamp())

def generate_upi_id(merchant: str) -> str:
    """Generate a realistic UPI ID for a merchant."""
    clean_name = merchant.lower().replace(" ", "").replace("'", "")[:10]
    return f"{clean_name}{random.choice(UPI_SUFFIXES)}"

def generate_analysis_json(category: str, amount: float, decision: str) -> str:
    """Generate mock analysis JSON for prepay checks. Must include final_category and final_confidence for summary/spend-by-category/confidence-trend."""
    risk_score = 0.1 if decision == "allow" else (0.5 if decision == "warn" else 0.9)
    confidence = round(random.uniform(0.75, 0.95), 2)

    analysis = {
        "category": category,
        "final_category": category,
        "confidence": confidence,
        "final_confidence": confidence,
        "risk_score": round(risk_score + random.uniform(-0.05, 0.05), 2),
        "policy_rules_applied": ["amount_check", "merchant_allowlist", "category_policy"],
        "llm_reasoning": f"Transaction categorized as {category}. {'Normal expense.' if decision == 'allow' else 'Requires attention.' if decision == 'warn' else 'Policy violation detected.'}",
        "risk_flags": {
            "high_amount": amount > 50000,
            "unverified_merchant": False,
            "low_quality_note": False,
            "policy_violation": decision in ("warn", "block"),
        },
        "subscription": {"is_subscription": False, "confidence": 0.0, "pattern": "none"},
        "fusion_result": {"label": category, "confidence": confidence, "model_used": "seed"},
        "policy_result": {"action": decision, "options": ["Continue", "Cancel"]},
    }
    return json.dumps(analysis)

def generate_splits_json(total: float) -> str:
    """Generate payment splits JSON."""
    categories = ["Food", "Transport", "Shopping", "Bills", "Entertainment"]
    num_splits = random.randint(2, 4)
    selected = random.sample(categories, num_splits)
    
    remaining = total
    splits = []
    for i, cat in enumerate(selected):
        if i == len(selected) - 1:
            amount = remaining
        else:
            amount = round(random.uniform(0.1, 0.4) * total, 2)
            remaining -= amount
        splits.append({"label": cat, "amount": round(amount, 2)})
    
    return json.dumps(splits)

# =============================================================================
# Data Generators
# =============================================================================

def generate_feedback_data() -> List[tuple]:
    """Generate feedback table entries."""
    data = []
    dates = get_dates_for_3_months()
    
    for _ in range(55):
        category = random.choice(list(MERCHANTS.keys()))
        merchant = random.choice(MERCHANTS[category])
        note = random.choice(NOTES_TEMPLATES[category])
        text = f"{merchant} - {note}"
        correct_label = category
        user_id = "user-123" if random.random() < 0.4 else random.choice(USER_IDS)
        created_at = random_timestamp_for_date(random.choice(dates))
        data.append((text, correct_label, user_id, created_at))
    
    return data

def generate_prepay_checks_data() -> List[tuple]:
    """Generate prepay_checks table entries with patterns."""
    data = []
    dates = get_dates_for_3_months()
    
    # Weekly pattern: Coffee Shop every Monday
    mondays = [d for d in dates if d.weekday() == 0]
    for monday in mondays[:12]:
        data.append(create_prepay_entry(
            "Starbucks", "Food", 450.0, "weekly coffee meeting", monday, "allow"
        ))
    
    # Weekly pattern: Uber every Friday
    fridays = [d for d in dates if d.weekday() == 4]
    for friday in fridays[:12]:
        data.append(create_prepay_entry(
            "Uber", "Transport", random.uniform(200, 600), "office commute", friday, "allow"
        ))
    
    # Monthly patterns: Bills on specific days
    for month_offset in range(3):
        base = datetime.now() - timedelta(days=30 * month_offset)
        
        # Electricity on 5th
        elec_date = base.replace(day=5)
        data.append(create_prepay_entry(
            "Electricity Board", "Utilities", random.uniform(2000, 3500), 
            "monthly electricity bill", elec_date, "allow"
        ))
        
        # WiFi on 10th
        wifi_date = base.replace(day=10)
        data.append(create_prepay_entry(
            "Jio Fiber", "Utilities", 999.0, 
            "monthly wifi bill", wifi_date, "allow"
        ))
        
        # Rent on 1st
        rent_date = base.replace(day=1)
        data.append(create_prepay_entry(
            "Landlord", "Business", 25000.0, 
            "monthly rent", rent_date, "allow"
        ))
    
    # Regular varied transactions (weight user-123 so frontend default user sees data)
    for i in range(50):
        category = random.choice(list(MERCHANTS.keys()))
        merchant = random.choice(MERCHANTS[category])
        note = random.choice(NOTES_TEMPLATES[category])
        amount = round(random.uniform(100, 5000), 2)
        date = random.choice(dates)
        decision = random.choice(["allow", "allow", "allow", "warn"])
        uid = "user-123" if random.random() < 0.45 else None
        data.append(create_prepay_entry(merchant, category, amount, note, date, decision, user_id=uid))
    
    # High amount transactions (for alert testing)
    high_amounts = [45000, 52000, 68000, 75000, 48000, 55000]
    high_merchants = ["Hotel Taj", "Conference Center", "Hotel Marriott", "Airport Lounge", "Business Travel", "Equipment Purchase"]
    high_notes = ["annual conference", "client summit", "business accommodation", "executive travel", "equipment procurement", "team offsite"]
    
    for i, amount in enumerate(high_amounts):
        date = random.choice(dates)
        data.append(create_prepay_entry(
            high_merchants[i], "Business", float(amount), 
            high_notes[i], date, random.choice(["warn", "block"])
        ))
    
    # Blocked transactions (policy violations)
    blocked_items = [
        ("Wine Shop", "Entertainment", 3500.0, "team celebration"),
        ("Casino", "Entertainment", 15000.0, "client entertainment"),
        ("Luxury Spa", "Entertainment", 8000.0, "personal expense"),
    ]
    for merchant, cat, amount, note in blocked_items:
        date = random.choice(dates)
        data.append(create_prepay_entry(merchant, cat, amount, note, date, "block"))
    
    return data

def create_prepay_entry(merchant: str, category: str, amount: float, note: str,
                        date: datetime, decision: str, user_id: Optional[str] = None) -> tuple:
    """Helper to create a prepay check entry tuple. Pass user_id to force (e.g. user-123 for demo)."""
    if user_id is None:
        user_id = random.choice(USER_IDS)
    upi_id = generate_upi_id(merchant)
    user_role = random.choice(USER_ROLES)
    date_str = date.strftime("%Y-%m-%d")
    analysis_json = generate_analysis_json(category, amount, decision)
    created_at = random_timestamp_for_date(date)
    
    return (user_id, merchant, amount, note, upi_id, user_role, date_str, decision, analysis_json, created_at)

def generate_payment_splits_data() -> List[tuple]:
    """Generate payment_splits table entries."""
    data = []
    dates = get_dates_for_3_months()
    
    totals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7500, 8000, 9000, 10000, 12000, 15000]

    for i, total in enumerate(totals):
        user_id = "user-123" if random.random() < 0.5 else random.choice(USER_IDS)
        splits_json = generate_splits_json(float(total))
        created_at = random_timestamp_for_date(random.choice(dates))
        data.append((user_id, float(total), splits_json, created_at))
    
    return data

def generate_subscriptions_data() -> List[tuple]:
    """Generate subscriptions table entries."""
    data = []
    base_date = datetime.now() - timedelta(days=60)  # Started 2 months ago
    
    for sub in SUBSCRIPTIONS:
        user_id = "user-123" if random.random() < 0.5 else random.choice(USER_IDS[:3])
        created_at = random_timestamp_for_date(base_date - timedelta(days=random.randint(0, 30)))
        data.append((user_id, sub["name"], sub["amount"], sub["period"], created_at))
    
    return data

# =============================================================================
# Database Operations
# =============================================================================

def clear_all_data(conn: sqlite3.Connection):
    """Clear all data from all tables."""
    cursor = conn.cursor()
    tables = ["feedback", "prepay_checks", "payment_splits", "subscriptions"]
    
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        print(f"  ✓ Cleared {table}")
    
    conn.commit()

def insert_feedback(conn: sqlite3.Connection, data: List[tuple]):
    """Insert feedback data."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO feedback (text, correct_label, user_id, created_at) VALUES (?, ?, ?, ?)",
        data
    )
    conn.commit()
    print(f"  ✓ Inserted {len(data)} feedback records")

def insert_prepay_checks(conn: sqlite3.Connection, data: List[tuple]):
    """Insert prepay_checks data."""
    cursor = conn.cursor()
    cursor.executemany(
        """INSERT INTO prepay_checks 
        (user_id, merchant, amount, note, upi_id, user_role, date, decision, analysis_json, created_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        data
    )
    conn.commit()
    print(f"  ✓ Inserted {len(data)} prepay_checks records")

def insert_payment_splits(conn: sqlite3.Connection, data: List[tuple]):
    """Insert payment_splits data."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO payment_splits (user_id, total_amount, splits_json, created_at) VALUES (?, ?, ?, ?)",
        data
    )
    conn.commit()
    print(f"  ✓ Inserted {len(data)} payment_splits records")

def insert_subscriptions(conn: sqlite3.Connection, data: List[tuple]):
    """Insert subscriptions data."""
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT INTO subscriptions (user_id, name, amount, period, created_at) VALUES (?, ?, ?, ?, ?)",
        data
    )
    conn.commit()
    print(f"  ✓ Inserted {len(data)} subscription records")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("\n" + "="*60)
    print("CalcBERT v2 - Seed Data Generator")
    print("="*60)
    print(f"Database: {DB_PATH}")
    print(f"Date Range: {(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
    print()
    
    # Initialize database (creates tables if not exist)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Create tables if they don't exist
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        correct_label TEXT NOT NULL,
        user_id TEXT,
        created_at INTEGER NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prepay_checks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        merchant TEXT NOT NULL,
        amount REAL NOT NULL,
        note TEXT,
        upi_id TEXT,
        user_role TEXT NOT NULL,
        date TEXT NOT NULL,
        decision TEXT NOT NULL,
        analysis_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS payment_splits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        total_amount REAL NOT NULL,
        splits_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        amount REAL NOT NULL,
        period TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    conn.commit()
    
    # Step 1: Clear existing data
    print("Step 1: Clearing existing data...")
    clear_all_data(conn)
    print()
    
    # Step 2: Generate and insert new data
    print("Step 2: Generating and inserting seed data...")
    
    feedback_data = generate_feedback_data()
    insert_feedback(conn, feedback_data)
    
    prepay_data = generate_prepay_checks_data()
    insert_prepay_checks(conn, prepay_data)
    
    splits_data = generate_payment_splits_data()
    insert_payment_splits(conn, splits_data)
    
    subs_data = generate_subscriptions_data()
    insert_subscriptions(conn, subs_data)
    
    print()
    
    # Step 3: Verification summary
    print("Step 3: Verification Summary")
    print("-" * 40)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM feedback")
    print(f"  feedback:        {cursor.fetchone()[0]} records")
    
    cursor.execute("SELECT COUNT(*) FROM prepay_checks")
    print(f"  prepay_checks:   {cursor.fetchone()[0]} records")
    
    cursor.execute("SELECT COUNT(*) FROM payment_splits")
    print(f"  payment_splits:  {cursor.fetchone()[0]} records")
    
    cursor.execute("SELECT COUNT(*) FROM subscriptions")
    print(f"  subscriptions:   {cursor.fetchone()[0]} records")
    
    # High amount check
    cursor.execute("SELECT COUNT(*) FROM prepay_checks WHERE amount > 40000")
    print(f"\n  High-value transactions (>₹40K): {cursor.fetchone()[0]}")
    
    # Pattern check
    cursor.execute("SELECT COUNT(*) FROM prepay_checks WHERE merchant = 'Starbucks'")
    print(f"  Weekly pattern (Starbucks):      {cursor.fetchone()[0]}")
    
    cursor.execute("SELECT COUNT(*) FROM prepay_checks WHERE merchant = 'Electricity Board'")
    print(f"  Monthly pattern (Electricity):   {cursor.fetchone()[0]}")
    
    conn.close()
    
    print()
    print("="*60)
    print("✅ Seed data generation complete!")
    print("="*60)
    print("\nRun 'py checkDB.py' to view the generated data.\n")

if __name__ == "__main__":
    main()
