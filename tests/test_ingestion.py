"""Unit and integration tests for document ingestion, extraction, chunking, and deduplication.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pymupdf as fitz
import pytest

from ingestion.chunker import SemanticChunker
from ingestion.metadata import calculate_file_hash, extract_pdf_metadata
from ingestion.pdf_loader import PDFLoader
from ingestion.text_extractor import PDFTextExtractor, PageData


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a temporary 3-page scientific PDF fixture with known content."""
    pdf_path = tmp_path / "test_transformer_paper.pdf"
    doc = fitz.open()

    p1 = doc.new_page(width=612, height=792)
    p1.insert_text(fitz.Point(54, 72), "Attention Is All You Need\nAshish Vaswani, Noam Shazeer\n1. Introduction\nWe propose the Transformer.", fontsize=14)

    p2 = doc.new_page(width=612, height=792)
    p2.insert_text(fitz.Point(54, 72), "2. Model Architecture\nThe encoder is composed of a stack of 6 identical layers. Self-attention complexity is O(n^2 * d).", fontsize=12)

    p3 = doc.new_page(width=612, height=792)
    p3.insert_text(fitz.Point(54, 72), "3. Results\nOur model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.", fontsize=12)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_pdf_validation_valid(sample_pdf: Path) -> None:
    loader = PDFLoader()
    is_valid, msg = loader.validate_pdf(sample_pdf)
    assert is_valid is True
    assert msg == "Valid PDF"


def test_pdf_validation_corrupted(tmp_path: Path) -> None:
    loader = PDFLoader()
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.5 this is not a valid pdf body")
    is_valid, msg = loader.validate_pdf(corrupt_file)
    assert is_valid is False


def test_pdf_validation_empty(tmp_path: Path) -> None:
    loader = PDFLoader()
    empty_file = tmp_path / "empty.pdf"
    empty_file.touch()
    is_valid, msg = loader.validate_pdf(empty_file)
    assert is_valid is False
    assert "0 bytes" in msg


def test_file_hash_and_metadata(sample_pdf: Path) -> None:
    file_hash = calculate_file_hash(sample_pdf)
    assert len(file_hash) == 64  # SHA-256

    loader = PDFLoader()
    info = loader.inspect_pdf(sample_pdf)
    assert info["valid"] is True
    assert info["total_pages"] == 3
    assert info["file_hash"] == file_hash


def test_text_extractor_page_numbering(sample_pdf: Path) -> None:
    extractor = PDFTextExtractor(enable_ocr=False)
    pages = extractor.extract_document(sample_pdf)

    assert len(pages) == 3
    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[2].page_number == 3

    assert "Attention Is All You Need" in pages[0].clean_text
    assert "Self-attention complexity" in pages[1].clean_text
    assert "28.4 BLEU" in pages[2].clean_text


def test_semantic_chunker_provenance() -> None:
    chunker = SemanticChunker(chunk_size_tokens=100, chunk_overlap_tokens=20, min_chunk_tokens=10)

    p1 = PageData(page_number=1, raw_text="Page 1 sentence. " * 10, clean_text="Page 1 sentence. " * 10, headings=["Intro"])
    p2 = PageData(page_number=2, raw_text="Page 2 sentence. " * 10, clean_text="Page 2 sentence. " * 10, headings=["Methods"])

    chunks = chunker.chunk_pages([p1, p2], document_id="doc123", document_title="Test Doc", document_filename="test.pdf")

    assert len(chunks) > 0
    for c in chunks:
        assert c.start_page >= 1
        assert c.end_page >= c.start_page
        assert c.document_id == "doc123"
        assert len(c.text) > 0
        assert c.token_count >= 10
