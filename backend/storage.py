import sqlite3
import time
from typing import List, Tuple, Optional
import os

DB_PATH = "backend/backend_feedback.db"


def init_db() -> None:
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create feedback table
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        correct_label TEXT NOT NULL,
        user_id TEXT,
        created_at INTEGER NOT NULL
    )
    """)
    
    # Create prepay_checks table for v2 backend
    c.execute("""
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
    
    # Create payment_splits table for split payment persistence
    c.execute("""
    CREATE TABLE IF NOT EXISTS payment_splits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        total_amount REAL NOT NULL,
        splits_json TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    
    # Create subscriptions table for user-managed subscriptions
    c.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        amount REAL NOT NULL,
        period TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )
    """)
    
    # Create alert_rules table for user-defined alert rules
    c.execute("""
    CREATE TABLE IF NOT EXISTS alert_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        category TEXT NOT NULL,
        limit_amount REAL NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at INTEGER NOT NULL
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")



def save_feedback(text: str, correct_label: str, user_id: Optional[str] = None) -> int:
    
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO feedback (text, correct_label, user_id, created_at) VALUES (?, ?, ?, ?)",
        (text, correct_label, user_id, ts)
    )
    fid = c.lastrowid
    conn.commit()
    conn.close()
    return fid


def get_feedback_samples(limit: Optional[int] = None) -> List[Tuple[int, str, str]]:
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    q = "SELECT id, text, correct_label FROM feedback ORDER BY created_at ASC"
    if limit:
        q += f" LIMIT {limit}"
    c.execute(q)
    rows = c.fetchall()
    conn.close()
    return rows


def get_feedback_count() -> int:
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback")
    count = c.fetchone()[0]
    conn.close()
    return count


def clear_feedback() -> None:
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM feedback")
    conn.commit()
    conn.close()
    print("All feedback cleared from database")



def get_recent_feedback(hours: int = 24) -> List[Tuple[int, str, str, int]]:
    
    cutoff = int(time.time()) - (hours * 3600)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, text, correct_label, created_at FROM feedback WHERE created_at >= ? ORDER BY created_at DESC",
        (cutoff,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ============================================================================
# v2 Backend: Prepay Check Storage Functions
# ============================================================================

def save_prepay_check(
    user_id: str,
    merchant: str,
    amount: float,
    note: str,
    upi_id: str,
    user_role: str,
    date: str,
    decision: str,
    analysis_json: str
) -> int:
    """
    Save a prepay check result to the database.
    
    Args:
        user_id: User identifier
        merchant: Merchant name
        amount: Transaction amount
        note: Transaction note/description
        upi_id: UPI ID for payment
        user_role: User role (employee, manager, admin)
        date: Transaction date
        decision: Policy decision (allow, warn, block, allow_with_note)
        analysis_json: JSON string of full analysis
        
    Returns:
        The ID of the inserted prepay check record
    """
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO prepay_checks 
        (user_id, merchant, amount, note, upi_id, user_role, date, decision, analysis_json, created_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, merchant, amount, note, upi_id, user_role, date, decision, analysis_json, ts)
    )
    check_id = c.lastrowid
    conn.commit()
    conn.close()
    return check_id


def get_prepay_checks(
    user_id: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Tuple]:
    """
    Retrieve prepay check records from the database.
    
    Args:
        user_id: Optional filter by user ID
        limit: Optional limit on number of records
        
    Returns:
        List of tuples with prepay check data
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        q = "SELECT * FROM prepay_checks WHERE user_id = ? ORDER BY created_at DESC"
        params = (user_id,)
    else:
        q = "SELECT * FROM prepay_checks ORDER BY created_at DESC"
        params = ()
    
    if limit:
        q += f" LIMIT {limit}"
    
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    return rows


def get_prepay_check_by_id(check_id: int) -> Optional[Tuple]:
    """
    Get a specific prepay check by ID.
    
    Args:
        check_id: The prepay check ID
        
    Returns:
        Tuple with prepay check data or None if not found
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM prepay_checks WHERE id = ?", (check_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_prepay_check_count(user_id: Optional[str] = None) -> int:
    """
    Get the total count of prepay checks.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        Number of prepay check records
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute("SELECT COUNT(*) FROM prepay_checks WHERE user_id = ?", (user_id,))
    else:
        c.execute("SELECT COUNT(*) FROM prepay_checks")
    
    count = c.fetchone()[0]
    conn.close()
    return count


# ============================================================================
# Payment Splits Storage Functions
# ============================================================================

def save_payment_split(user_id: str, total_amount: float, splits_json: str) -> int:
    """
    Save a payment split configuration to the database.
    
    Args:
        user_id: User identifier
        total_amount: Total amount to split
        splits_json: JSON string of split configuration [{"label": "Food", "amount": 200}, ...]
        
    Returns:
        The ID of the inserted split record
    """
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO payment_splits 
        (user_id, total_amount, splits_json, created_at) 
        VALUES (?, ?, ?, ?)""",
        (user_id, total_amount, splits_json, ts)
    )
    split_id = c.lastrowid
    conn.commit()
    conn.close()
    return split_id


def get_latest_payment_split(user_id: Optional[str] = None) -> Optional[Tuple]:
    """
    Get the most recent payment split configuration.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        Tuple with split data or None if not found
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute("SELECT * FROM payment_splits WHERE user_id = ? ORDER BY created_at DESC LIMIT 1", (user_id,))
    else:
        c.execute("SELECT * FROM payment_splits ORDER BY created_at DESC LIMIT 1")
    
    row = c.fetchone()
    conn.close()
    return row


def get_payment_splits_today(user_id: Optional[str] = None) -> List[Tuple]:
    """
    Get all payment splits created today.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        List of tuples with split data
    """
    from datetime import datetime as dt
    today = dt.now().date()
    today_start = int(dt.combine(today, dt.min.time()).timestamp())
    today_end = int(dt.combine(today, dt.max.time()).timestamp())
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute(
            "SELECT * FROM payment_splits WHERE user_id = ? AND created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
            (user_id, today_start, today_end)
        )
    else:
        c.execute(
            "SELECT * FROM payment_splits WHERE created_at >= ? AND created_at <= ? ORDER BY created_at DESC",
            (today_start, today_end)
        )
    
    rows = c.fetchall()
    conn.close()
    return rows


# ============================================================================
# Subscriptions Storage Functions
# ============================================================================

def save_subscription(user_id: str, name: str, amount: float, period: str) -> int:
    """
    Save a user subscription to the database.
    
    Args:
        user_id: User identifier
        name: Subscription name
        amount: Subscription amount
        period: Billing period (daily, weekly, monthly, yearly)
        
    Returns:
        The ID of the inserted subscription record
    """
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO subscriptions 
        (user_id, name, amount, period, created_at) 
        VALUES (?, ?, ?, ?, ?)""",
        (user_id, name, amount, period, ts)
    )
    sub_id = c.lastrowid
    conn.commit()
    conn.close()
    return sub_id


def get_user_subscriptions(user_id: Optional[str] = None) -> List[Tuple]:
    """
    Get all user subscriptions from the database.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        List of tuples with subscription data (id, user_id, name, amount, period, created_at)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute("SELECT * FROM subscriptions WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    else:
        c.execute("SELECT * FROM subscriptions ORDER BY created_at DESC")
    
    rows = c.fetchall()
    conn.close()
    return rows


def delete_subscription(subscription_id: int, user_id: Optional[str] = None) -> bool:
    """
    Delete a subscription from the database.
    
    Args:
        subscription_id: Subscription ID to delete
        user_id: Optional user ID for verification
        
    Returns:
        True if deleted, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute("DELETE FROM subscriptions WHERE id = ? AND user_id = ?", (subscription_id, user_id))
    else:
        c.execute("DELETE FROM subscriptions WHERE id = ?", (subscription_id,))
    
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ============================================================================
# Alert Rules Storage Functions
# ============================================================================

def get_feedback_for_text(text: str) -> Optional[Tuple[int, str, str, Optional[str], int]]:
    """
    Get the most recent feedback sample for an exact transaction text.

    Args:
        text: Transaction text used when saving feedback

    Returns:
        Tuple: (id, text, correct_label, user_id, created_at) or None
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, text, correct_label, user_id, created_at FROM feedback WHERE text = ? ORDER BY created_at DESC LIMIT 1",
        (text,)
    )
    row = c.fetchone()
    conn.close()
    return row


def save_alert_rule(user_id: str, category: str, limit_amount: float, enabled: bool = True) -> int:
    """
    Save an alert rule to the database.
    
    Args:
        user_id: User identifier
        category: Category to monitor
        limit_amount: Spending limit for the category
        enabled: Whether the rule is enabled
        
    Returns:
        The ID of the inserted alert rule record
    """
    ts = int(time.time())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """INSERT INTO alert_rules 
        (user_id, category, limit_amount, enabled, created_at) 
        VALUES (?, ?, ?, ?, ?)""",
        (user_id, category, limit_amount, 1 if enabled else 0, ts)
    )
    rule_id = c.lastrowid
    conn.commit()
    conn.close()
    return rule_id


def get_user_alert_rules(user_id: Optional[str] = None) -> List[Tuple]:
    """
    Get all alert rules for a user.
    
    Args:
        user_id: Optional filter by user ID
        
    Returns:
        List of tuples: (id, user_id, category, limit_amount, enabled, created_at)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute(
            "SELECT * FROM alert_rules WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        )
    else:
        c.execute("SELECT * FROM alert_rules ORDER BY created_at DESC")
    
    rules = c.fetchall()
    conn.close()
    return rules


def update_alert_rule(rule_id: int, enabled: Optional[bool] = None, limit_amount: Optional[float] = None, user_id: Optional[str] = None) -> bool:
    """
    Update an alert rule.
    
    Args:
        rule_id: Rule ID to update
        enabled: Optional new enabled status
        limit_amount: Optional new limit amount
        user_id: Optional user ID for verification
        
    Returns:
        True if updated, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    updates = []
    params = []
    
    if enabled is not None:
        updates.append("enabled = ?")
        params.append(1 if enabled else 0)
    
    if limit_amount is not None:
        updates.append("limit_amount = ?")
        params.append(limit_amount)
    
    if not updates:
        conn.close()
        return False
    
    params.append(rule_id)
    
    if user_id:
        params.append(user_id)
        query = f"UPDATE alert_rules SET {', '.join(updates)} WHERE id = ? AND user_id = ?"
    else:
        query = f"UPDATE alert_rules SET {', '.join(updates)} WHERE id = ?"
    
    c.execute(query, params)
    updated = c.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_alert_rule(rule_id: int, user_id: Optional[str] = None) -> bool:
    """
    Delete an alert rule.
    
    Args:
        rule_id: Rule ID to delete
        user_id: Optional user ID for verification
        
    Returns:
        True if deleted, False otherwise
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id:
        c.execute("DELETE FROM alert_rules WHERE id = ? AND user_id = ?", (rule_id, user_id))
    else:
        c.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
    
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

