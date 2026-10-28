"""FAISS vector index manager supporting incremental addition, deletion, and search.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from storage.index_repository import IndexRepository

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """Manages an IndexIDMap-wrapped IndexFlatIP for exact cosine/inner-product similarity search."""

    def __init__(
        self,
        dimension: int = 768,
        index_repo: Optional[IndexRepository] = None
    ) -> None:
        self.dimension = dimension
        self.index_repo = index_repo or IndexRepository()
        self.index: Optional[faiss.Index] = None
        self._initialize_or_load()

    def _initialize_or_load(self) -> None:
        """Load index from disk or create a new IndexIDMap2(IndexFlatIP)."""
        if self.index_repo.index_exists():
            try:
                loaded = self.index_repo.load_index()
                if loaded is not None and loaded.d == self.dimension:
                    self.index = loaded
                    logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors (dim={self.dimension})")
                    return
                else:
                    logger.warning("Loaded index dimension mismatch. Initializing new index.")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}. Initializing fresh index.")

        # Create new index
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)
        logger.info(f"Created new IndexFlatIP + IndexIDMap2 (dim={self.dimension})")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]) -> None:
        """Add vectors with custom integer IDs to support incremental ingestion."""
        if len(vectors) == 0:
            return

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors_np = np.ascontiguousarray(vectors, dtype=np.float32)
        ids_np = np.array(ids, dtype=np.int64)

        self.index.add_with_ids(vectors_np, ids_np)
        logger.info(f"Added {len(ids)} vectors to FAISS index. Total count: {self.index.ntotal}")

    def remove_ids(self, ids: List[int]) -> int:
        """Remove specific vector IDs from the index (supports document deletion)."""
        if not ids or self.index.ntotal == 0:
            return 0

        ids_selector = faiss.IDSelectorBatch(ids)
        removed = self.index.remove_ids(ids_selector)
        logger.info(f"Removed {removed} vectors from FAISS index. Total remaining: {self.index.ntotal}")
        return removed

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5
    ) -> Tuple[List[float], List[int]]:
        """Search top-k most similar vectors.

        Returns:
            Tuple of (similarity_scores, faiss_ids).
        """
        if self.index.ntotal == 0:
            return [], []

        # Prepare query vector shape (1, dimension)
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
        query_np = np.ascontiguousarray(query_vector, dtype=np.float32)

        actual_k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_np, actual_k)

        res_scores = [float(s) for s in scores[0] if s != -1]
        res_ids = [int(idx) for idx in indices[0] if idx != -1]

        return res_scores[:len(res_ids)], res_ids

    def save(self) -> None:
        """Persist index to disk."""
        meta = {
            "dimension": self.dimension,
            "total_vectors": self.index.ntotal if self.index else 0,
            "metric": "inner_product"
        }
        self.index_repo.save_index(self.index, meta)
        logger.info(f"Saved FAISS index to {self.index_repo.index_path}")

    def total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0

    def reset(self) -> None:
        """Clear all vectors and reset the index."""
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(base_index)
        self.index_repo.clear()
