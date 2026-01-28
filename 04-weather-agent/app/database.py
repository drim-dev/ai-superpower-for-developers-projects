import sqlite3
from contextlib import contextmanager
from app.config import DATABASE_PATH


@contextmanager
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                is_active INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        conn.commit()


def create_session() -> int:
    with get_db() as conn:
        cursor = conn.execute("INSERT INTO sessions DEFAULT VALUES")
        conn.commit()
        return cursor.lastrowid


def get_session(session_id: int) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def get_all_sessions() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def end_session(session_id: int) -> bool:
    with get_db() as conn:
        cursor = conn.execute(
            "UPDATE sessions SET is_active = 0, ended_at = CURRENT_TIMESTAMP WHERE id = ? AND is_active = 1",
            (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def add_message(session_id: int, role: str, content: str) -> int:
    with get_db() as conn:
        cursor = conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        return cursor.lastrowid


def get_session_messages(session_id: int) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]
