"""Document metadata extraction and hashing utilities.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pymupdf as fitz


def calculate_file_hash(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_pdf_metadata(doc: fitz.Document, file_path: Path | str) -> Dict[str, Any]:
    """Extract document title, authors, publication year, and basic metadata."""
    meta = doc.metadata or {}
    filename = Path(file_path).name

    # 1. Title extraction
    title = meta.get("title")
    if not title or title.strip().lower() in ["untitled", "microsoft word", "none"]:
        title = _extract_title_from_first_page(doc) or filename.rsplit(".", 1)[0].replace("_", " ").title()
    else:
        title = title.strip()

    # 2. Author extraction
    authors_str = meta.get("author") or ""
    authors: List[str] = [a.strip() for a in re.split(r"[,;]", authors_str) if a.strip()]
    if not authors:
        authors = _extract_authors_heuristic(doc)

    # 3. Publication Year extraction
    year = _extract_year(meta, doc)

    return {
        "filename": filename,
        "title": title,
        "authors": authors,
        "publication_year": year,
        "total_pages": len(doc),
        "file_size_bytes": Path(file_path).stat().st_size if Path(file_path).exists() else 0,
        "format": meta.get("format", "PDF"),
        "producer": meta.get("producer", "")
    }


def _extract_title_from_first_page(doc: fitz.Document) -> Optional[str]:
    """Heuristic: extract the largest font text in the upper half of page 1."""
    if len(doc) == 0:
        return None
    page = doc[0]
    blocks = page.get_text("dict").get("blocks", [])

    candidates = []
    page_height = page.rect.height

    for block in blocks:
        if block.get("type") != 0:  # text block
            continue
        bbox = block.get("bbox", [0, 0, 0, 0])
        # Look in top 60% of first page
        if bbox[1] > page_height * 0.6:
            continue

        for line in block.get("lines", []):
            line_text = ""
            max_size = 0.0
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_text += (" " if line_text else "") + text
                    max_size = max(max_size, span.get("size", 0.0))
            if line_text and len(line_text) > 5 and max_size > 11.0:
                candidates.append((max_size, -bbox[1], line_text))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        # Take the candidate with maximum font size
        best_title = candidates[0][2]
        # Clean up whitespace
        best_title = re.sub(r"\s+", " ", best_title).strip()
        if len(best_title) > 8:
            return best_title[:200]
    return None


def _extract_authors_heuristic(doc: fitz.Document) -> List[str]:
    """Extract candidate author names from below title on page 1."""
    if len(doc) == 0:
        return []
    text = doc[0].get_text("text")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    # Typically authors appear within the first 10-15 lines
    for line in lines[1:8]:
        if re.search(r"\b(abstract|introduction|keywords|department|university|email)\b", line, re.I):
            break
        # If line contains names separated by comma or 'and'
        if 5 < len(line) < 120 and not re.search(r"\d{3,}", line):
            parts = re.split(r",|\band\b", line)
            clean_parts = [p.strip() for p in parts if len(p.strip().split()) in (2, 3, 4)]
            if clean_parts:
                return clean_parts[:5]
    return []


def _extract_year(meta: Dict[str, Any], doc: fitz.Document) -> Optional[int]:
    """Extract 4-digit publication year from metadata or first page text."""
    # Check creationDate e.g. "D:20230514..."
    creation_date = meta.get("creationDate", "")
    if creation_date:
        match = re.search(r"20\d\d|19\d\d", creation_date)
        if match:
            year = int(match.group(0))
            if 1950 <= year <= 2030:
                return year

    # Search in first page header/footer
    if len(doc) > 0:
        first_page_text = doc[0].get_text("text")
        matches = re.findall(r"\b(19\d\d|20\d\d)\b", first_page_text[:1500])
        for m in matches:
            year = int(m)
            if 1970 <= year <= 2026:
                return year
    return None
