"""Metadata filters for query refinement and subset searching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MetadataFilter:
    document_ids: Optional[List[str]] = None
    filenames: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    section_keywords: Optional[List[str]] = None

    def matches(self, chunk: Dict[str, Any], doc_meta: Optional[Dict[str, Any]] = None) -> bool:
        """Check whether a chunk and its parent document pass the filter conditions."""
        if self.document_ids and chunk.get("document_id") not in self.document_ids:
            return False

        if self.filenames and chunk.get("document_filename") not in self.filenames:
            return False

        if self.section_keywords:
            heading = (chunk.get("section_heading") or "").lower()
            if not any(kw.lower() in heading for kw in self.section_keywords):
                return False

        if doc_meta:
            year = doc_meta.get("publication_year")
            if self.min_year is not None and (year is None or year < self.min_year):
                return False
            if self.max_year is not None and (year is None or year > self.max_year):
                return False

            if self.authors:
                doc_authors = str(doc_meta.get("authors") or "").lower()
                if not any(auth.lower() in doc_authors for auth in self.authors):
                    return False

        return True
