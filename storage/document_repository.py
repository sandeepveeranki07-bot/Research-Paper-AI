"""Repository pattern for Document, Page, and Chunk persistence in SQLite.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from storage.database import get_db


class DocumentRepository:
    def __init__(self) -> None:
        pass

    def insert_document(self, doc: Dict[str, Any]) -> str:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO documents (
                    id, filename, title, authors, publication_year,
                    total_pages, file_hash, file_path, file_size_bytes,
                    ingestion_status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc["id"],
                doc["filename"],
                doc.get("title", doc["filename"]),
                json.dumps(doc.get("authors", [])) if isinstance(doc.get("authors"), list) else doc.get("authors"),
                doc.get("publication_year"),
                doc.get("total_pages", 0),
                doc["file_hash"],
                doc["file_path"],
                doc.get("file_size_bytes", 0),
                doc.get("ingestion_status", "indexed"),
                doc.get("error_message")
            ))
            return doc["id"]

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_documents(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            if search:
                cursor.execute(
                    "SELECT * FROM documents WHERE title LIKE ? OR filename LIKE ? ORDER BY created_at DESC",
                    (f"%{search}%", f"%{search}%")
                )
            else:
                cursor.execute("SELECT * FROM documents ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, doc_id: str) -> bool:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            return cursor.rowcount > 0

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO chunks (
                    chunk_id, document_id, document_title, document_filename,
                    chunk_index, text, token_count, start_page, end_page,
                    section_heading, faiss_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c["chunk_id"],
                    c["document_id"],
                    c.get("document_title"),
                    c["document_filename"],
                    c["chunk_index"],
                    c["text"],
                    c.get("token_count", 0),
                    c["start_page"],
                    c["end_page"],
                    c.get("section_heading"),
                    c.get("faiss_id")
                ) for c in chunks
            ])

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_chunks_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC", (doc_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_chunks_by_faiss_ids(self, faiss_ids: List[int]) -> List[Dict[str, Any]]:
        if not faiss_ids:
            return []
        placeholders = ",".join("?" for _ in faiss_ids)
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM chunks WHERE faiss_id IN ({placeholders})", faiss_ids)
            rows = cursor.fetchall()
            mapping = {row["faiss_id"]: dict(row) for row in rows}
            # Return in the exact order requested
            return [mapping[fid] for fid in faiss_ids if fid in mapping]

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks ORDER BY faiss_id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_document_stats(self) -> Dict[str, Any]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS doc_count, COALESCE(SUM(total_pages), 0) AS total_pages FROM documents WHERE ingestion_status = 'indexed'")
            doc_stats = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) AS chunk_count FROM chunks")
            chunk_stats = cursor.fetchone()

            cursor.execute("SELECT COUNT(*) AS failure_count FROM documents WHERE ingestion_status = 'failed'")
            fail_stats = cursor.fetchone()

            return {
                "total_documents": doc_stats["doc_count"] if doc_stats else 0,
                "total_pages": doc_stats["total_pages"] if doc_stats else 0,
                "total_chunks": chunk_stats["chunk_count"] if chunk_stats else 0,
                "ingestion_failures": fail_stats["failure_count"] if fail_stats else 0
            }

    def get_next_faiss_id(self) -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(faiss_id), -1) + 1 AS next_id FROM chunks")
            row = cursor.fetchone()
            return int(row["next_id"]) if row else 0
