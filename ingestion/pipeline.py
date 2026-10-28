"""End-to-end PDF ingestion pipeline orchestrating extraction, chunking, embedding, and storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np

from config import get_settings
from ingestion.chunker import SemanticChunker, TextChunk
from ingestion.metadata import calculate_file_hash
from ingestion.pdf_loader import PDFLoader
from ingestion.text_extractor import PDFTextExtractor
from retrieval.embeddings import EmbeddingEngine
from retrieval.faiss_index import FAISSIndexManager
from storage.database import get_db, init_database
from storage.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    successful_docs: int = 0
    failed_docs: int = 0
    skipped_duplicates: int = 0
    total_pages_processed: int = 0
    total_chunks_indexed: int = 0
    errors: List[Dict[str, str]] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Ingestion complete: {self.successful_docs} indexed, {self.skipped_duplicates} duplicate(s) skipped, "
            f"{self.failed_docs} failed. Processed {self.total_pages_processed} pages, {self.total_chunks_indexed} chunks."
        )


class IngestionPipeline:
    def __init__(
        self,
        loader: Optional[PDFLoader] = None,
        extractor: Optional[PDFTextExtractor] = None,
        chunker: Optional[SemanticChunker] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        faiss_manager: Optional[FAISSIndexManager] = None,
        doc_repo: Optional[DocumentRepository] = None
    ) -> None:
        settings = get_settings()
        init_database()

        self.loader = loader or PDFLoader()
        self.extractor = extractor or PDFTextExtractor(
            enable_ocr=settings.enable_ocr_fallback,
            tesseract_cmd=settings.tesseract_cmd
        )
        self.chunker = chunker or SemanticChunker(
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
            min_chunk_tokens=settings.min_chunk_tokens
        )
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.faiss_manager = faiss_manager or FAISSIndexManager(dimension=self.embedding_engine.get_dimension())
        self.doc_repo = doc_repo or DocumentRepository()

    def ingest_files(
        self,
        file_paths: List[Path | str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> IngestionResult:
        """Ingest multiple PDF files incrementally with deduplication, chunking, and indexing."""
        result = IngestionResult()
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths):
            path = Path(file_path)
            if progress_callback:
                progress_callback(idx + 1, total_files, f"Processing {path.name}...")

            try:
                # 1. Validation and inspection
                info = self.loader.inspect_pdf(path)
                if not info["valid"]:
                    result.failed_docs += 1
                    result.errors.append({"file": path.name, "error": info.get("error", "Validation failed")})
                    self._record_failed_document(path, info.get("error", "Validation failed"))
                    continue

                file_hash = info["file_hash"]

                # 2. Duplicate detection
                existing = self.doc_repo.get_document_by_hash(file_hash)
                if existing and existing.get("ingestion_status") == "indexed":
                    logger.info(f"Duplicate document detected (Hash: {file_hash[:8]}). Skipping: {path.name}")
                    result.skipped_duplicates += 1
                    result.document_ids.append(existing["id"])
                    continue

                # 3. Text & Structure extraction
                pages = self.extractor.extract_document(path)
                if not pages:
                    err_msg = "No readable text or pages extracted."
                    result.failed_docs += 1
                    result.errors.append({"file": path.name, "error": err_msg})
                    self._record_failed_document(path, err_msg, file_hash)
                    continue

                doc_id = file_hash[:16]
                doc_title = info.get("title") or path.stem.replace("_", " ").title()

                # 4. Semantic chunking with page-level provenance
                chunks = self.chunker.chunk_pages(
                    pages=pages,
                    document_id=doc_id,
                    document_title=doc_title,
                    document_filename=path.name
                )

                if not chunks:
                    err_msg = "Extracted text was insufficient to generate semantic chunks."
                    result.failed_docs += 1
                    result.errors.append({"file": path.name, "error": err_msg})
                    self._record_failed_document(path, err_msg, file_hash)
                    continue

                # 5. Embedding generation
                chunk_texts = [c.text for c in chunks]
                embeddings = self.embedding_engine.embed_texts(chunk_texts, show_progress=False)

                # 6. Assign monotonic FAISS IDs
                start_faiss_id = self.doc_repo.get_next_faiss_id()
                faiss_ids = [start_faiss_id + i for i in range(len(chunks))]

                # 7. Add vectors to FAISS index incrementally
                self.faiss_manager.add_vectors(embeddings, faiss_ids)

                # 8. Record in SQLite
                doc_record = {
                    "id": doc_id,
                    "filename": path.name,
                    "title": doc_title,
                    "authors": info.get("authors", []),
                    "publication_year": info.get("publication_year"),
                    "total_pages": len(pages),
                    "file_hash": file_hash,
                    "file_path": str(path.resolve()),
                    "file_size_bytes": info.get("file_size_bytes", 0),
                    "ingestion_status": "indexed",
                    "error_message": None
                }
                self.doc_repo.insert_document(doc_record)

                # Insert pages
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.executemany("""
                        INSERT OR REPLACE INTO document_pages (
                            document_id, page_number, text_length, has_images, is_ocr
                        ) VALUES (?, ?, ?, ?, ?)
                    """, [(doc_id, p.page_number, len(p.clean_text), p.has_images, p.is_ocr) for p in pages])

                # Insert chunks with metadata and assigned faiss_id
                chunk_dicts = []
                for c, fid in zip(chunks, faiss_ids):
                    chunk_dicts.append({
                        "chunk_id": c.chunk_id,
                        "document_id": doc_id,
                        "document_title": doc_title,
                        "document_filename": path.name,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "token_count": c.token_count,
                        "start_page": c.start_page,
                        "end_page": c.end_page,
                        "section_heading": c.section_heading,
                        "faiss_id": fid
                    })
                self.doc_repo.insert_chunks(chunk_dicts)

                # 9. Persist FAISS index after successful document addition
                self.faiss_manager.save()

                result.successful_docs += 1
                result.total_pages_processed += len(pages)
                result.total_chunks_indexed += len(chunks)
                result.document_ids.append(doc_id)

            except Exception as e:
                logger.exception(f"Unexpected error ingesting {path.name}: {e}")
                result.failed_docs += 1
                result.errors.append({"file": path.name, "error": str(e)})
                self._record_failed_document(path, str(e))

        return result

    def ingest_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> IngestionResult:
        """Scan directory and ingest all discovered PDFs."""
        pdfs = self.loader.discover_pdfs(directory, recursive=recursive)
        logger.info(f"Discovered {len(pdfs)} PDF(s) in {directory}")
        return self.ingest_files(pdfs, progress_callback=progress_callback)

    def delete_document(self, doc_id: str) -> bool:
        """Delete document from database and purge its vectors from FAISS index."""
        chunks = self.doc_repo.get_chunks_by_document(doc_id)
        faiss_ids = [c["faiss_id"] for c in chunks if c.get("faiss_id") is not None]

        # Remove vectors from FAISS
        if faiss_ids:
            self.faiss_manager.remove_ids(faiss_ids)
            self.faiss_manager.save()

        # Delete document from SQLite (foreign keys cascade to chunks and pages)
        deleted = self.doc_repo.delete_document(doc_id)
        logger.info(f"Deleted document {doc_id} (status: {deleted}, removed {len(faiss_ids)} vectors)")
        return deleted

    def rebuild_index(self) -> int:
        """Reconstruct entire FAISS index from stored chunks in SQLite."""
        logger.info("Reconstructing FAISS index from SQLite chunks...")
        self.faiss_manager.reset()

        all_chunks = self.doc_repo.get_all_chunks()
        if not all_chunks:
            self.faiss_manager.save()
            return 0

        texts = [c["text"] for c in all_chunks]
        ids = [c["faiss_id"] for c in all_chunks]

        embeddings = self.embedding_engine.embed_texts(texts, show_progress=True)
        self.faiss_manager.add_vectors(embeddings, ids)
        self.faiss_manager.save()
        logger.info(f"Reconstructed FAISS index with {len(ids)} vectors.")
        return len(ids)

    def _record_failed_document(self, path: Path, error_msg: str, file_hash: str = "") -> None:
        """Store record of failed document ingestion for monitoring."""
        try:
            h = file_hash or (calculate_file_hash(path) if path.exists() and path.stat().st_size > 0 else f"fail_{path.name}")
            with get_db() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        id, filename, title, total_pages, file_hash, file_path,
                        ingestion_status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    h[:16], path.name, path.name, 0, h, str(path.resolve()),
                    "failed", error_msg
                ))
        except Exception as e:
            logger.error(f"Failed to record failure for {path.name}: {e}")
