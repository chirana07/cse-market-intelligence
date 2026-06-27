from __future__ import annotations

import sqlite3
from pathlib import Path

from src.settings import SETTINGS


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_type TEXT,
    title TEXT,
    domain TEXT,
    primary_ticker TEXT,
    primary_event TEXT,
    ingested_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT,
    source_url TEXT NOT NULL,
    chunk_index INTEGER,
    text_hash TEXT,
    snippet TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_source_url ON documents(source_url);
CREATE INDEX IF NOT EXISTS idx_documents_ticker ON documents(primary_ticker);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    event_type TEXT,
    logged_at TEXT NOT NULL,
    analysis_mode TEXT,
    ticker TEXT,
    company_name TEXT,
    latency_sec REAL,
    critic_status TEXT,
    confidence TEXT,
    evidence_score INTEGER,
    source_count INTEGER,
    chunk_count INTEGER,
    request_json TEXT NOT NULL DEFAULT '{}',
    evidence_json TEXT NOT NULL DEFAULT '{}',
    critic_json TEXT NOT NULL DEFAULT '{}',
    output_validation_json TEXT NOT NULL DEFAULT '{}',
    trajectory_json TEXT NOT NULL DEFAULT '[]',
    sources_json TEXT NOT NULL DEFAULT '[]',
    raw_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_logged_at ON agent_runs(logged_at);
CREATE INDEX IF NOT EXISTS idx_agent_runs_ticker ON agent_runs(ticker);
CREATE INDEX IF NOT EXISTS idx_agent_runs_critic ON agent_runs(critic_status);
"""

MIGRATION_SQL = [
    "ALTER TABLE agent_runs ADD COLUMN output_validation_json TEXT NOT NULL DEFAULT '{}'",
]


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else SETTINGS.sqlite_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        for statement in MIGRATION_SQL:
            try:
                conn.execute(statement)
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
