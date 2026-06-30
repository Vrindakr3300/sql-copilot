"""
Safety gate — classifies SQL statements as read-only or mutating,
and provides a dry-run mechanism so mutating queries can be tested
inside a rolled-back transaction before the user confirms execution.
"""

from dataclasses import dataclass
from enum import Enum

import sqlparse
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Statement types that mutate data or schema
_MUTATING_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "MERGE", "GRANT", "REVOKE",
}


class QuerySafety(str, Enum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    UNKNOWN = "unknown"


@dataclass
class SafetyCheckResult:
    safety: QuerySafety
    statement_type: str
    reason: str


def classify_query(sql: str) -> SafetyCheckResult:
    """Classifies a SQL string as read-only or mutating using sqlparse."""
    parsed = sqlparse.parse(sql.strip())

    if not parsed:
        return SafetyCheckResult(QuerySafety.UNKNOWN, "EMPTY", "Could not parse query.")

    statement = parsed[0]
    stmt_type = statement.get_type()  # e.g. 'SELECT', 'INSERT', 'UNKNOWN'

    if stmt_type == "SELECT" or stmt_type == "UNKNOWN" and sql.strip().upper().startswith("WITH"):
        return SafetyCheckResult(QuerySafety.READ_ONLY, stmt_type, "Read-only query.")

    first_token = sql.strip().split()[0].upper() if sql.strip() else ""

    if first_token in _MUTATING_KEYWORDS:
        return SafetyCheckResult(
            QuerySafety.MUTATING,
            first_token,
            f"Query begins with mutating keyword '{first_token}'.",
        )

    if stmt_type == "SELECT":
        return SafetyCheckResult(QuerySafety.READ_ONLY, stmt_type, "Read-only query.")

    return SafetyCheckResult(
        QuerySafety.UNKNOWN,
        stmt_type,
        f"Could not confidently classify statement type '{stmt_type}' — treat as unsafe.",
    )


def dry_run(engine: Engine, sql: str) -> tuple[bool, str]:
    """
    Executes a mutating query inside a transaction, then rolls it back —
    proves the query is valid without actually committing changes.
    Returns (success, message).
    """
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                conn.execute(text(sql))
                trans.rollback()
                return True, "Dry run successful — query is valid. No changes were committed."
            except Exception as e:
                trans.rollback()
                return False, f"Dry run failed: {e}"
    except Exception as e:
        return False, f"Could not establish transaction: {e}"


def execute_query(engine: Engine, sql: str, safety: QuerySafety) -> tuple[bool, object]:
    """
    Executes a query for real. For READ_ONLY, returns fetched rows.
    For MUTATING, commits the transaction and returns rowcount.
    Caller is responsible for confirming with the user before calling
    this for a MUTATING query.
    """
    try:
        with engine.connect() as conn:
            if safety == QuerySafety.READ_ONLY:
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result]
                return True, rows
            else:
                trans = conn.begin()
                result = conn.execute(text(sql))
                trans.commit()
                return True, result.rowcount
    except Exception as e:
        return False, str(e)