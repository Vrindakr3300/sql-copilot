"""
Memory layer — sliding-window conversation history for the current
session, plus a persisted SQLite log of every query run and its outcome.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

_MEMORY_DB_PATH = Path("session_memory.db")
_MAX_HISTORY_TURNS = 6  # sliding window size (user+agent pairs)


class SessionMemory:
    def __init__(self):
        self.history: list[dict] = []  # in-memory sliding window for this run
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(_MEMORY_DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                question TEXT,
                sql TEXT,
                success INTEGER,
                result_summary TEXT
            )
        """)
        conn.commit()
        conn.close()

    def add_turn(self, question: str, answer: str):
        """Adds a user question + agent answer pair to the sliding window."""
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > _MAX_HISTORY_TURNS:
            self.history.pop(0)

    def get_history_text(self) -> str:
        """Formats the sliding window as text to prepend to the next prompt."""
        if not self.history:
            return ""
        lines = ["Previous conversation in this session:"]
        for turn in self.history:
            lines.append(f"User asked: {turn['question']}")
            lines.append(f"You answered: {turn['answer']}")
        return "\n".join(lines)

    def log_query(self, question: str, sql: str, success: bool, result_summary: str):
        """Persists a query attempt to the SQLite log, regardless of session."""
        conn = sqlite3.connect(_MEMORY_DB_PATH)
        conn.execute(
            "INSERT INTO query_log (timestamp, question, sql, success, result_summary) VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), question, sql, int(success), result_summary[:500]),
        )
        conn.commit()
        conn.close()

    def get_recent_log(self, limit: int = 10) -> list[dict]:
        """Returns the most recent persisted query log entries."""
        conn = sqlite3.connect(_MEMORY_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]