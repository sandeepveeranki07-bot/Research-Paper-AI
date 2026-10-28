"""Index repository for FAISS persistence, loading, and synchronization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional
import faiss
from config import get_settings


class IndexRepository:
    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ) -> None:
        settings = get_settings()
        self.index_path = index_path or settings.faiss_index_path
        self.metadata_path = metadata_path or settings.faiss_metadata_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def save_index(self, index: faiss.Index, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist FAISS index binary and metadata JSON to disk."""
        faiss.write_index(index, str(self.index_path))
        if metadata is not None:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

    def load_index(self) -> Optional[faiss.Index]:
        """Load persisted FAISS index if it exists on disk."""
        if not self.index_path.exists():
            return None
        return faiss.read_index(str(self.index_path))

    def load_metadata(self) -> Dict[str, Any]:
        """Load index metadata (dimension, count, metric, etc.)."""
        if not self.metadata_path.exists():
            return {}
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def index_exists(self) -> bool:
        return self.index_path.exists()

    def clear(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
