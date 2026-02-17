"""SQLite database for local operational state.

Tracks processed messages, source polling cursors, and local audit log.
All operations are idempotent and safe to retry.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from sdr.config import DB_PATH


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db(db_path: Path = DB_PATH):
    """Context manager for SQLite connections."""
    _ensure_dir(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with get_db(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS processed_messages (
                source TEXT NOT NULL,
                source_message_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'processed',
                attempts INTEGER NOT NULL DEFAULT 1,
                error TEXT,
                airtable_message_id TEXT,
                airtable_contact_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(source, source_message_id)
            );

            CREATE TABLE IF NOT EXISTS source_state (
                source TEXT PRIMARY KEY,
                last_poll_at TEXT,
                cursor TEXT,
                gmail_history_id TEXT
            );

            CREATE TABLE IF NOT EXISTS local_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                trace_id TEXT,
                action TEXT NOT NULL,
                source TEXT,
                message_id TEXT,
                contact_id TEXT,
                details TEXT,
                duration_ms INTEGER
            );

            CREATE TABLE IF NOT EXISTS learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                deactivated_at TEXT
            );
        """)


def is_message_processed(source: str, source_message_id: str, db_path: Path = DB_PATH) -> bool:
    """Check if a message has already been processed."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM processed_messages WHERE source = ? AND source_message_id = ?",
            (source, source_message_id),
        ).fetchone()
        return row is not None


def mark_message_processed(
    source: str,
    source_message_id: str,
    status: str = "processed",
    airtable_message_id: Optional[str] = None,
    airtable_contact_id: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Mark a message as processed (upsert)."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO processed_messages
               (source, source_message_id, status, airtable_message_id, airtable_contact_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(source, source_message_id) DO UPDATE SET
                 status = excluded.status,
                 airtable_message_id = COALESCE(excluded.airtable_message_id, airtable_message_id),
                 airtable_contact_id = COALESCE(excluded.airtable_contact_id, airtable_contact_id),
                 updated_at = datetime('now')""",
            (source, source_message_id, status, airtable_message_id, airtable_contact_id),
        )


def mark_message_failed(
    source: str, source_message_id: str, error: str, db_path: Path = DB_PATH
) -> None:
    """Mark a message as failed, incrementing attempt count."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO processed_messages
               (source, source_message_id, status, error)
               VALUES (?, ?, 'failed', ?)
               ON CONFLICT(source, source_message_id) DO UPDATE SET
                 status = 'failed',
                 error = excluded.error,
                 attempts = attempts + 1,
                 updated_at = datetime('now')""",
            (source, source_message_id, error),
        )


def get_failed_messages(
    max_attempts: int = 3, db_path: Path = DB_PATH
) -> list[dict]:
    """Get messages that failed but haven't exceeded retry limit."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            """SELECT source, source_message_id, attempts, error
               FROM processed_messages
               WHERE status = 'failed' AND attempts < ?""",
            (max_attempts,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_source_state(source: str, db_path: Path = DB_PATH) -> Optional[dict]:
    """Get polling state for a source."""
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM source_state WHERE source = ?", (source,)
        ).fetchone()
        return dict(row) if row else None


def update_source_state(
    source: str,
    cursor: Optional[str] = None,
    gmail_history_id: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Update polling state for a source."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO source_state (source, last_poll_at, cursor, gmail_history_id)
               VALUES (?, datetime('now'), ?, ?)
               ON CONFLICT(source) DO UPDATE SET
                 last_poll_at = datetime('now'),
                 cursor = COALESCE(excluded.cursor, cursor),
                 gmail_history_id = COALESCE(excluded.gmail_history_id, gmail_history_id)""",
            (source, cursor, gmail_history_id),
        )


def get_active_learned_rules(db_path: Path = DB_PATH) -> list[dict]:
    """Return all active learned rules, ordered by creation date."""
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, rule_text, confidence, created_at FROM learning_log WHERE active = 1 ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]


def insert_learned_rule(rule_text: str, confidence: float, db_path: Path = DB_PATH) -> int:
    """Insert a new learned rule and return its id."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO learning_log (rule_text, confidence) VALUES (?, ?)",
            (rule_text, confidence),
        )
        return cursor.lastrowid


def deactivate_learned_rule(rule_id: int, db_path: Path = DB_PATH) -> None:
    """Deactivate a learned rule by id."""
    with get_db(db_path) as conn:
        conn.execute(
            "UPDATE learning_log SET active = 0, deactivated_at = datetime('now') WHERE id = ?",
            (rule_id,),
        )


def log_local_audit(
    action: str,
    trace_id: Optional[str] = None,
    source: Optional[str] = None,
    message_id: Optional[str] = None,
    contact_id: Optional[str] = None,
    details: Optional[dict] = None,
    duration_ms: Optional[int] = None,
    db_path: Path = DB_PATH,
) -> None:
    """Write an entry to the local audit log."""
    with get_db(db_path) as conn:
        conn.execute(
            """INSERT INTO local_audit
               (trace_id, action, source, message_id, contact_id, details, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                trace_id,
                action,
                source,
                message_id,
                contact_id,
                json.dumps(details) if details else None,
                duration_ms,
            ),
        )
