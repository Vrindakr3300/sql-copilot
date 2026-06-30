"""
Schema introspection — extracts table, column, and foreign key
metadata from a connected database for use in prompts and embeddings.
"""

from dataclasses import dataclass, field
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool = False


@dataclass
class ForeignKeyInfo:
    column: str
    references_table: str
    references_column: str


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    sample_rows: list[dict] = field(default_factory=list)


def get_table_info(engine: Engine, table_name: str, sample_limit: int = 3) -> TableInfo:
    """Builds a full TableInfo object for a single table."""
    inspector = inspect(engine)

    pk_columns = set(inspector.get_pk_constraint(table_name).get("constrained_columns", []))

    columns = [
        ColumnInfo(
            name=col["name"],
            type=str(col["type"]),
            nullable=col["nullable"],
            primary_key=col["name"] in pk_columns,
        )
        for col in inspector.get_columns(table_name)
    ]

    foreign_keys = [
        ForeignKeyInfo(
            column=fk["constrained_columns"][0],
            references_table=fk["referred_table"],
            references_column=fk["referred_columns"][0],
        )
        for fk in inspector.get_foreign_keys(table_name)
        if fk.get("constrained_columns") and fk.get("referred_columns")
    ]

    sample_rows = []
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {sample_limit}"))
            sample_rows = [dict(row._mapping) for row in result]
    except Exception:
        # Some DBs (e.g. SQL Server) don't support LIMIT — fail silently for now
        pass

    return TableInfo(name=table_name, columns=columns, foreign_keys=foreign_keys, sample_rows=sample_rows)


def get_full_schema(engine: Engine) -> list[TableInfo]:
    """Builds TableInfo for every table in the database."""
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    return [get_table_info(engine, name) for name in table_names]


def table_to_text_chunk(table: TableInfo) -> str:
    """
    Converts a TableInfo into a natural-language text chunk —
    this is what gets embedded for the RAG layer later.
    """
    col_descriptions = ", ".join(
        f"{c.name} ({c.type}{', PK' if c.primary_key else ''})" for c in table.columns
    )
    fk_descriptions = "; ".join(
        f"{fk.column} -> {fk.references_table}.{fk.references_column}" for fk in table.foreign_keys
    )

    chunk = f"Table '{table.name}' has columns: {col_descriptions}."
    if fk_descriptions:
        chunk += f" Foreign keys: {fk_descriptions}."
    if table.sample_rows:
        chunk += f" Sample row: {table.sample_rows[0]}."

    return chunk