"""
SQLite-based chat history storage.
Each user session gets its own set of records identified by a session_id.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chat_history.db')


def init_db():
    """Create the chat_history table if it does not already exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                timestamp   TEXT    NOT NULL,
                user_message TEXT   NOT NULL,
                bot_response TEXT   NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_session ON chat_history (session_id)')
        conn.commit()


def save_message(session_id: str, user_message: str, bot_response: str):
    """Persist one user/bot exchange."""
    timestamp = datetime.utcnow().isoformat(sep=' ', timespec='seconds')
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            'INSERT INTO chat_history (session_id, timestamp, user_message, bot_response) VALUES (?, ?, ?, ?)',
            (session_id, timestamp, user_message, bot_response),
        )
        conn.commit()


def get_history(session_id: str, limit: int = 20):
    """Return the most recent *limit* exchanges for a session, oldest first."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            '''SELECT timestamp, user_message, bot_response
               FROM chat_history
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT ?''',
            (session_id, limit),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def clear_history(session_id: str):
    """Delete all records for a session."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
        conn.commit()
