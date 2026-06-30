"""
Database connection layer — provides a unified interface across
Postgres, MySQL, and SQL Server using SQLAlchemy.
"""

from dataclasses import dataclass
from enum import Enum
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


class DBType(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MSSQL = "mssql"


@dataclass
class ConnectionConfig:
    db_type: DBType
    host: str
    port: int
    username: str
    password: str
    database: str


def build_connection_string(config: ConnectionConfig) -> str:
    """Builds the correct SQLAlchemy connection string per DB type."""
    if config.db_type == DBType.POSTGRES:
        return (
            f"postgresql+psycopg2://{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )
    elif config.db_type == DBType.MYSQL:
        return (
            f"mysql+pymysql://{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
        )
    elif config.db_type == DBType.MSSQL:
        return (
            f"mssql+pyodbc://{config.username}:{config.password}"
            f"@{config.host}:{config.port}/{config.database}"
            f"?driver=ODBC+Driver+17+for+SQL+Server"
        )
    else:
        raise ValueError(f"Unsupported db_type: {config.db_type}")


def get_engine(config: ConnectionConfig) -> Engine:
    """Creates and returns a SQLAlchemy engine for the given config."""
    conn_str = build_connection_string(config)
    return create_engine(conn_str, pool_pre_ping=True)


def test_connection(engine: Engine) -> bool:
    """Runs a trivial query to confirm the connection works."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


def list_tables(engine: Engine) -> list[str]:
    """Returns all table names visible to the connected user."""
    inspector = inspect(engine)
    return inspector.get_table_names()