"""
SQLite storage for feedback data.
Provides functions to initialize database, save feedback, and retrieve samples.
"""

import sqlite3
import time
from typing import List, Tuple, Optional
import os

DB_PATH = "backend/backend_feedback.db"


def init_db() -> None:
    """
    Initialize the feedback database.
    Creates the feedback table if it doesn't exist.
    """
    # Ensure backend directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        correct_label TEXT NOT NULL,
        user_id TEXT,
        created_at INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def save_feedback(text: str, correct_label: str, user_id: Optional[str] = None) -> int:
    """
    Save user feedback to the database.
    
    Args:
        text: The transaction text
        correct_label: The correct category label
        user_id: Optional user identifier
        
    Returns:
        The ID of the inserted feedback record
    """
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
    """
    Retrieve feedback samples from the database.
    
    Args:
        limit: Optional limit on number of samples to retrieve
        
    Returns:
        List of tuples (id, text, correct_label)
    """
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
    """
    Get the total count of feedback samples.
    
    Returns:
        Number of feedback records in the database
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM feedback")
    count = c.fetchone()[0]
    conn.close()
    return count


def clear_feedback() -> None:
    """
    Clear all feedback from the database.
    Use with caution - this deletes all stored feedback.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM feedback")
    conn.commit()
    conn.close()
    print("All feedback cleared from database")


def get_recent_feedback(hours: int = 24) -> List[Tuple[int, str, str, int]]:
    """
    Get feedback from the last N hours.
    
    Args:
        hours: Number of hours to look back
        
    Returns:
        List of tuples (id, text, correct_label, created_at)
    """
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
