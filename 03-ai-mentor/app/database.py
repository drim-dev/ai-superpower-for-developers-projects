"""Database operations for sessions and messages."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from app.config import DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize the database with required tables."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        conn.commit()


def create_session() -> int:
    """Create a new mentoring session and return its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions DEFAULT VALUES")
        conn.commit()
        return cursor.lastrowid


def end_session(session_id: int) -> bool:
    """End an active session. Returns True if successful."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET is_active = 0, ended_at = ? WHERE id = ? AND is_active = 1",
            (datetime.now(), session_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_session(session_id: int) -> dict | None:
    """Get session by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def add_message(session_id: int, role: str, content: str) -> int:
    """Add a message to a session and return its ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        return cursor.lastrowid


def get_session_messages(session_id: int) -> list[dict]:
    """Get all messages for a session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_all_sessions() -> list[dict]:
    """Get all sessions."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]
