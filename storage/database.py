"""SQLite database connection and schema management for Research Intelligence Assistant.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from config import get_settings

_local = threading.local()


def get_db_path() -> Path:
    settings = get_settings()
    settings.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings.sqlite_db_path


def get_connection() -> sqlite3.Connection:
    """Get a thread-local SQLite connection with foreign keys enabled."""
    db_path = get_db_path()
    conn = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for acquiring and closing/committing a database connection."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Initialize database schema, tables, and indices."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            title TEXT,
            authors TEXT,
            publication_year INTEGER,
            total_pages INTEGER NOT NULL DEFAULT 0,
            file_hash TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL DEFAULT 0,
            ingestion_status TEXT NOT NULL DEFAULT 'indexed',
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Document pages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            text_length INTEGER DEFAULT 0,
            has_images BOOLEAN DEFAULT 0,
            is_ocr BOOLEAN DEFAULT 0,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
            UNIQUE(document_id, page_number)
        );
        """)

        # Chunks table (with precise page provenance and section headings)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            document_title TEXT,
            document_filename TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            start_page INTEGER NOT NULL,
            end_page INTEGER NOT NULL,
            section_heading TEXT,
            faiss_id INTEGER UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """)

        # Indices for rapid querying and filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(file_hash);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_year ON documents(publication_year);")

        # Evaluation runs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_questions INTEGER NOT NULL,
            recall_at_5 REAL,
            citation_accuracy REAL,
            citation_coverage REAL,
            avg_retrieval_latency_ms REAL,
            avg_generation_latency_ms REAL,
            report_json TEXT
        );
        """)
