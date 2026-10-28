"""Scanned PDF page detection and OCR fallback processing.
"""

from __future__ import annotations

import logging
from typing import Optional
import pymupdf as fitz

logger = logging.getLogger(__name__)


def is_scanned_page(page: fitz.Page, min_chars: int = 40) -> bool:
    """Determine if a PDF page appears to be a scanned document without embedded text.

    Args:
        page: fitz.Page object.
        min_chars: Minimum character count below which an image-heavy page is deemed scanned.
    """
    text = page.get_text("text").strip()
    if len(text) >= min_chars:
        return False

    # Check if page contains images
    images = page.get_images(full=True)
    if len(images) > 0:
        return True

    # Check for raster drawings or xobjects
    drawings = page.get_drawings()
    if not text and (len(drawings) > 0 or len(images) > 0):
        return True

    return False


def ocr_page(
    page: fitz.Page,
    tesseract_cmd: Optional[str] = None,
    dpi: int = 200
) -> str:
    """Perform OCR on a page using PyMuPDF pixmap and pytesseract when available.

    Falls back gracefully if pytesseract or Tesseract-OCR is not installed.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))

        text = pytesseract.image_to_string(image)
        logger.info(f"OCR successfully extracted {len(text)} characters from page {page.number + 1}")
        return text.strip()

    except ImportError:
        logger.warning(
            "pytesseract or PIL is not installed. Scanned page OCR is unavailable. "
            "Please install pytesseract and tesseract-ocr to enable OCR fallback."
        )
        return ""
    except Exception as e:
        logger.warning(f"OCR failed for page {page.number + 1}: {e}")
        return ""
