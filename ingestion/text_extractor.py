"""Robust PDF text extraction using PyMuPDF with header/footer removal and heading detection.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pymupdf as fitz

from ingestion.ocr import is_scanned_page, ocr_page

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    page_number: int  # 1-indexed
    raw_text: str
    clean_text: str
    headings: List[str] = field(default_factory=list)
    is_ocr: bool = False
    has_images: bool = False
    char_count: int = 0


class PDFTextExtractor:
    def __init__(
        self,
        enable_ocr: bool = True,
        tesseract_cmd: Optional[str] = None
    ) -> None:
        self.enable_ocr = enable_ocr
        self.tesseract_cmd = tesseract_cmd

    def extract_document(self, doc_or_path: fitz.Document | str | Path) -> List[PageData]:
        """Extract text and metadata from all pages in a PDF document.

        Handles malformed pages, scans, repeated headers/footers, and heading detection.
        """
        doc = None
        should_close = False

        if isinstance(doc_or_path, (str, Path)):
            doc = fitz.open(str(doc_or_path))
            should_close = True
        else:
            doc = doc_or_path

        try:
            if doc.is_encrypted:
                raise ValueError("PDF is encrypted and password-protected.")

            total_pages = len(doc)
            if total_pages == 0:
                logger.warning("PDF contains 0 pages.")
                return []

            # Step 1: Detect recurring headers and footers across pages
            repeated_lines = self._identify_repeated_headers_footers(doc)

            # Step 2: Extract text and structure page by page
            extracted_pages: List[PageData] = []
            current_active_heading = ""

            for p_idx in range(total_pages):
                page = doc[p_idx]
                page_num = p_idx + 1

                # Check for scanned page
                scanned = is_scanned_page(page)
                raw_text = ""
                used_ocr = False

                if scanned and self.enable_ocr:
                    raw_text = ocr_page(page, tesseract_cmd=self.tesseract_cmd)
                    used_ocr = bool(raw_text)
                else:
                    raw_text = page.get_text("text")

                # Extract headings on this page
                headings = self._detect_headings(page)
                if headings:
                    current_active_heading = headings[-1]

                # Clean text: remove repeated headers/footers and normalize spaces
                clean_text = self._clean_page_text(raw_text, repeated_lines)

                images = page.get_images(full=True)
                has_imgs = len(images) > 0

                page_data = PageData(
                    page_number=page_num,
                    raw_text=raw_text,
                    clean_text=clean_text,
                    headings=headings if headings else ([current_active_heading] if current_active_heading else []),
                    is_ocr=used_ocr,
                    has_images=has_imgs,
                    char_count=len(clean_text)
                )
                extracted_pages.append(page_data)

            return extracted_pages

        finally:
            if should_close and doc is not None:
                doc.close()

    def _identify_repeated_headers_footers(self, doc: fitz.Document) -> set[str]:
        """Find lines in top 10% and bottom 10% that appear across >= 3 pages."""
        header_footer_counts: Dict[str, int] = {}
        total_pages = len(doc)
        if total_pages < 3:
            return set()

        for page in doc:
            page_height = page.rect.height
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:
                    continue
                bbox = block.get("bbox", [0, 0, 0, 0])
                # Check top 8% (header) or bottom 8% (footer)
                if bbox[1] < page_height * 0.08 or bbox[3] > page_height * 0.92:
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        line_text = line_text.strip()
                        # Ignore pure numbers (page numbers) since those change per page
                        if line_text and not line_text.isdigit() and len(line_text) > 4:
                            header_footer_counts[line_text] = header_footer_counts.get(line_text, 0) + 1

        # Any line appearing in 3+ pages is classified as recurring header/footer
        return {line for line, count in header_footer_counts.items() if count >= 3}

    def _detect_headings(self, page: fitz.Page) -> List[str]:
        """Detect section headings on a page using font size, style, and regex."""
        headings: List[str] = []
        blocks = page.get_text("dict").get("blocks", [])

        # Estimate average body font size
        font_sizes: List[float] = []
        for block in blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if len(span.get("text", "").strip()) > 3:
                            font_sizes.append(span.get("size", 10.0))

        median_size = 10.0
        if font_sizes:
            font_sizes.sort()
            median_size = font_sizes[len(font_sizes) // 2]

        heading_pattern = re.compile(
            r"^((\d+(\.\d+)*\s+)?(abstract|introduction|related\s+work|background|methodology|methods|system\s+design|architecture|experiments?|results?|discussion|conclusions?|references|acknowledg(e)?ments?)\b)",
            re.IGNORECASE
        )

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text or len(line_text) > 100:
                    continue

                max_span_size = max((span.get("size", 10.0) for span in line.get("spans", [])), default=10.0)
                is_bold = any(("bold" in span.get("font", "").lower() or span.get("flags", 0) & 2 != 0) for span in line.get("spans", []))

                is_larger = max_span_size >= median_size * 1.15
                matches_regex = bool(heading_pattern.match(line_text))

                if (is_larger and is_bold) or matches_regex:
                    clean_heading = re.sub(r"\s+", " ", line_text).strip()
                    if clean_heading not in headings:
                        headings.append(clean_heading)

        return headings

    def _clean_page_text(self, text: str, repeated_lines: set[str]) -> str:
        """Strip recurring headers/footers and normalize excessive whitespace."""
        lines = text.split("\n")
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in repeated_lines:
                continue
            # Remove isolated standalone page numbers
            if re.match(r"^\d{1,4}$", stripped):
                continue
            filtered_lines.append(line)

        cleaned = "\n".join(filtered_lines)
        # Normalize double newlines and horizontal spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
