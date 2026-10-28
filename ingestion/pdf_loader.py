"""PDF loader and directory scanner with validation and error resilience.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pymupdf as fitz

from ingestion.metadata import calculate_file_hash, extract_pdf_metadata

logger = logging.getLogger(__name__)


class PDFLoader:
    def __init__(self) -> None:
        pass

    def validate_pdf(self, file_path: Path | str) -> Tuple[bool, str]:
        """Validate if a file is an uncorrupted, readable PDF."""
        path = Path(file_path)
        if not path.exists():
            return False, f"File does not exist: {path}"
        if not path.is_file():
            return False, f"Path is not a regular file: {path}"
        if path.stat().st_size == 0:
            return False, "File is empty (0 bytes)."

        # Check PDF magic bytes '%PDF-'
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                if header != b"%PDF-":
                    return False, f"Invalid PDF header: {header!r}"
        except Exception as e:
            return False, f"Unable to read file bytes: {e}"

        # Attempt to open with PyMuPDF
        try:
            doc = fitz.open(str(path))
            if doc.is_encrypted:
                doc.close()
                return False, "PDF is encrypted/password-protected."
            total_pages = len(doc)
            doc.close()
            if total_pages == 0:
                return False, "PDF has 0 pages."
            return True, "Valid PDF"
        except Exception as e:
            return False, f"Corrupted or invalid PDF format: {e}"

    def inspect_pdf(self, file_path: Path | str) -> Dict[str, Any]:
        """Inspect PDF basic metadata and hash."""
        path = Path(file_path)
        is_valid, msg = self.validate_pdf(path)
        if not is_valid:
            return {
                "valid": False,
                "error": msg,
                "filename": path.name,
                "file_path": str(path),
                "file_hash": calculate_file_hash(path) if path.exists() and path.is_file() and path.stat().st_size > 0 else "",
                "total_pages": 0
            }

        try:
            file_hash = calculate_file_hash(path)
            doc = fitz.open(str(path))
            metadata = extract_pdf_metadata(doc, path)
            doc.close()

            metadata["valid"] = True
            metadata["error"] = None
            metadata["file_hash"] = file_hash
            metadata["file_path"] = str(path)
            return metadata
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "filename": path.name,
                "file_path": str(path),
                "file_hash": "",
                "total_pages": 0
            }

    def discover_pdfs(self, directory: Path | str, recursive: bool = True) -> List[Path]:
        """Discover all PDF files in a directory."""
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted([p for p in dir_path.glob(pattern) if p.is_file()])
