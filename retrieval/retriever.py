"""Semantic retriever combining query embedding, FAISS similarity search, and SQLite metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from config import get_settings
from retrieval.embeddings import EmbeddingEngine
from retrieval.faiss_index import FAISSIndexManager
from retrieval.filters import MetadataFilter
from storage.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_title: str
    document_filename: str
    start_page: int
    end_page: int
    text: str
    similarity_score: float  # Uncalibrated inner-product / cosine similarity [-1.0, 1.0]
    section_heading: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_year: Optional[int] = None
    file_path: Optional[str] = None

    @property
    def page_citation(self) -> str:
        """Formatted page citation e.g. 'Page 3' or 'Pages 3-4'."""
        if self.start_page == self.end_page:
            return f"p. {self.start_page}"
        return f"pp. {self.start_page}-{self.end_page}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_filename": self.document_filename,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_citation": self.page_citation,
            "section_heading": self.section_heading,
            "text": self.text,
            "similarity_score": round(self.similarity_score, 4),
            "authors": self.authors,
            "publication_year": self.publication_year,
            "file_path": self.file_path
        }


class SemanticRetriever:
    """Retrieval coordinator that queries FAISS and joins SQLite chunk metadata."""

    def __init__(
        self,
        embedding_engine: Optional[EmbeddingEngine] = None,
        faiss_manager: Optional[FAISSIndexManager] = None,
        doc_repo: Optional[DocumentRepository] = None
    ) -> None:
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.doc_repo = doc_repo or DocumentRepository()
        self.faiss_manager = faiss_manager or FAISSIndexManager(dimension=self.embedding_engine.get_dimension())

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[MetadataFilter] = None,
        min_score: Optional[float] = None
    ) -> List[RetrievedChunk]:
        """Retrieve most relevant chunks for a given natural-language query."""
        settings = get_settings()
        k = top_k if top_k is not None else settings.default_top_k
        threshold = min_score if min_score is not None else settings.retrieval_similarity_threshold

        if not query or not query.strip():
            return []

        # If filters are present, retrieve a larger candidate pool to ensure top-k matches survive
        fetch_k = k * 4 if filters else k

        query_vec = self.embedding_engine.embed_query(query)
        scores, faiss_ids = self.faiss_manager.search(query_vec, k=fetch_k)

        if not faiss_ids:
            return []

        # Fetch chunk metadata from SQLite
        raw_chunks = self.doc_repo.get_chunks_by_faiss_ids(faiss_ids)
        chunk_by_fid = {c["faiss_id"]: c for c in raw_chunks}

        # Cache document metadata to avoid repeated queries
        doc_cache: Dict[str, Dict[str, Any]] = {}

        results: List[RetrievedChunk] = []
        for score, fid in zip(scores, faiss_ids):
            if score < threshold:
                continue

            chunk = chunk_by_fid.get(fid)
            if not chunk:
                continue

            doc_id = chunk["document_id"]
            if doc_id not in doc_cache:
                doc = self.doc_repo.get_document_by_id(doc_id)
                doc_cache[doc_id] = doc or {}
            doc_meta = doc_cache[doc_id]

            # Apply metadata filters
            if filters and not filters.matches(chunk, doc_meta):
                continue

            # Parse authors
            authors_raw = doc_meta.get("authors")
            authors: List[str] = []
            if authors_raw:
                try:
                    authors = json.loads(authors_raw) if isinstance(authors_raw, str) and authors_raw.startswith("[") else [authors_raw]
                except Exception:
                    authors = [str(authors_raw)]

            retrieved = RetrievedChunk(
                chunk_id=chunk["chunk_id"],
                document_id=doc_id,
                document_title=chunk.get("document_title") or doc_meta.get("title") or chunk["document_filename"],
                document_filename=chunk["document_filename"],
                start_page=chunk["start_page"],
                end_page=chunk["end_page"],
                text=chunk["text"],
                similarity_score=score,
                section_heading=chunk.get("section_heading"),
                authors=authors,
                publication_year=doc_meta.get("publication_year"),
                file_path=doc_meta.get("file_path")
            )
            results.append(retrieved)

            if len(results) >= k:
                break

        return results
