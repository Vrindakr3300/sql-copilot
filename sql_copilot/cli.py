"""
CLI entry point for sql-copilot.
"""

import os
import sqlite3
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import create_engine

from sql_copilot.db import ConnectionConfig, DBType, get_engine, test_connection, list_tables
from sql_copilot.schema import get_full_schema, table_to_text_chunk

app = typer.Typer(help="SQL Co-Pilot — agentic, schema-aware SQL assistant")
console = Console()

DEMO_DB_PATH = Path("demo.db")


@app.command()
def setup_demo():
    """Creates a small local SQLite demo database with sample data — no Docker needed."""
    if DEMO_DB_PATH.exists():
        DEMO_DB_PATH.unlink()

    conn = sqlite3.connect(DEMO_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            total REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    cur.executemany(
        "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
        [(1, "Asha Rao", "asha@example.com"), (2, "Ben Lee", "ben@example.com")],
    )
    cur.executemany(
        "INSERT INTO orders (id, customer_id, total) VALUES (?, ?, ?)",
        [(1, 1, 250.0), (2, 1, 75.5), (3, 2, 120.0)],
    )

    conn.commit()
    conn.close()
    console.print(f"[green]Demo database created at {DEMO_DB_PATH.resolve()}[/green]")


@app.command()
def connect_demo():
    """Connects to the local demo SQLite database and lists its tables."""
    if not DEMO_DB_PATH.exists():
        console.print("[red]Demo database not found. Run 'setup-demo' first.[/red]")
        raise typer.Exit(1)

    engine = create_engine(f"sqlite:///{DEMO_DB_PATH}")

    if test_connection(engine):
        console.print("[green]Connection successful.[/green]")
    else:
        console.print("[red]Connection failed.[/red]")
        raise typer.Exit(1)

    tables = list_tables(engine)
    table_display = Table(title="Tables found")
    table_display.add_column("Name", style="cyan")
    for t in tables:
        table_display.add_row(t)
    console.print(table_display)


@app.command()
def inspect_demo():
    """Prints the full schema of the demo database as text chunks (RAG preview)."""
    if not DEMO_DB_PATH.exists():
        console.print("[red]Demo database not found. Run 'setup-demo' first.[/red]")
        raise typer.Exit(1)

    engine = create_engine(f"sqlite:///{DEMO_DB_PATH}")
    schema = get_full_schema(engine)

    for table in schema:
        chunk = table_to_text_chunk(table)
        console.print(f"[bold cyan]{table.name}[/bold cyan]")
        console.print(chunk)
        console.print()


@app.command()
def test_retrieval(question: str):
    """Indexes the demo schema and retrieves relevant tables for a question."""
    if not DEMO_DB_PATH.exists():
        console.print("[red]Demo database not found. Run 'setup-demo' first.[/red]")
        raise typer.Exit(1)

    from sql_copilot.embeddings import SchemaRetriever

    engine = create_engine(f"sqlite:///{DEMO_DB_PATH}")
    schema = get_full_schema(engine)

    console.print("[cyan]Indexing schema...[/cyan]")
    retriever = SchemaRetriever()
    retriever.index_schema(schema)

    console.print(f"[cyan]Retrieving tables relevant to:[/cyan] {question}")
    relevant_chunks = retriever.retrieve_relevant_tables(question)

    for chunk in relevant_chunks:
        console.print(f"- {chunk}")

@app.command()
def ask(question: str):
    """Asks a natural-language question and gets back a generated SQL query."""
    if not DEMO_DB_PATH.exists():
        console.print("[red]Demo database not found. Run 'setup-demo' first.[/red]")
        raise typer.Exit(1)

    from sql_copilot.embeddings import SchemaRetriever
    from sql_copilot.agent import QueryGenerator

    engine = create_engine(f"sqlite:///{DEMO_DB_PATH}")
    schema = get_full_schema(engine)

    retriever = SchemaRetriever()
    retriever.index_schema(schema)
    relevant_chunks = retriever.retrieve_relevant_tables(question)

    console.print("[cyan]Generating SQL...[/cyan]")
    generator = QueryGenerator()
    sql = generator.generate_sql(question, relevant_chunks)

    console.print("[bold green]Generated SQL:[/bold green]")
    console.print(sql)

if __name__ == "__main__":
    app()