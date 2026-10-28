"""Unit tests for citation validation, hallucination detection, and unsupported-answer handling.
"""

from __future__ import annotations

import pytest

from generation.answer_schema import CitationClaim, RAGResponse
from generation.citation_validator import CitationValidator
from generation.rag_pipeline import RAGPipeline
from retrieval.retriever import RetrievedChunk
from storage.document_repository import DocumentRepository


@pytest.fixture
def mock_retrieved_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk_001",
            document_id="doc_attn",
            document_title="Attention Is All You Need",
            document_filename="attention.pdf",
            start_page=3,
            end_page=4,
            text="Scaled Dot-Product Attention is computed as softmax(QK^T / sqrt(d_k))V. Self-attention complexity is O(n^2 * d).",
            similarity_score=0.88,
            section_heading="3.2 Attention"
        )
    ]


def test_citation_validator_valid_claim(mock_retrieved_chunks, monkeypatch) -> None:
    # Mock document in repo
    class MockRepo:
        def list_documents(self):
            return [{"id": "doc_attn", "title": "Attention Is All You Need", "filename": "attention.pdf", "total_pages": 8}]

    validator = CitationValidator(doc_repo=MockRepo())

    valid_claim = CitationClaim(
        claim_text="Self-attention complexity is O(n^2 * d)",
        source_document="Attention Is All You Need",
        source_page=3,
        supporting_evidence="Self-attention complexity is O(n^2 * d).",
        evidence_relevance="direct"
    )

    validated = validator.validate_claims([valid_claim], mock_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].is_valid is True
    assert validated[0].validation_status == "valid"


def test_citation_validator_invalid_document(mock_retrieved_chunks) -> None:
    class MockRepo:
        def list_documents(self):
            return [{"id": "doc_attn", "title": "Attention Is All You Need", "filename": "attention.pdf", "total_pages": 8}]

    validator = CitationValidator(doc_repo=MockRepo())

    hallucinated_claim = CitationClaim(
        claim_text="Quantum computing solves attention in O(1)",
        source_document="Nonexistent Fabricated Paper 2026",
        source_page=1,
        supporting_evidence="Quantum computation accelerates attention.",
        evidence_relevance="direct"
    )

    validated = validator.validate_claims([hallucinated_claim], mock_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].is_valid is False
    assert validated[0].validation_status == "invalid_document"


def test_citation_validator_invalid_page(mock_retrieved_chunks) -> None:
    class MockRepo:
        def list_documents(self):
            return [{"id": "doc_attn", "title": "Attention Is All You Need", "filename": "attention.pdf", "total_pages": 8}]

    validator = CitationValidator(doc_repo=MockRepo())

    out_of_bounds_claim = CitationClaim(
        claim_text="Scaled dot product attention formula",
        source_document="Attention Is All You Need",
        source_page=99,  # Paper only has 8 pages!
        supporting_evidence="Scaled dot product attention formula.",
        evidence_relevance="direct"
    )

    validated = validator.validate_claims([out_of_bounds_claim], mock_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].is_valid is False
    assert validated[0].validation_status == "invalid_page"


def test_citation_validator_unsupported_passage(mock_retrieved_chunks) -> None:
    class MockRepo:
        def list_documents(self):
            return [{"id": "doc_attn", "title": "Attention Is All You Need", "filename": "attention.pdf", "total_pages": 8}]

    validator = CitationValidator(doc_repo=MockRepo())

    unsupported_claim = CitationClaim(
        claim_text="Transformers use biological synaptic plasticity algorithms",
        source_document="Attention Is All You Need",
        source_page=3,
        supporting_evidence="Biological synaptic plasticity algorithms.",
        evidence_relevance="direct"
    )

    validated = validator.validate_claims([unsupported_claim], mock_retrieved_chunks)
    assert len(validated) == 1
    assert validated[0].is_valid is False
    assert validated[0].validation_status == "unsupported_passage"


def test_unsupported_answer_handling() -> None:
    # Test RAG pipeline when no relevant evidence exists
    class MockEmptyRetriever:
        def retrieve(self, query, top_k=5, filters=None, min_score=None):
            return []

    pipeline = RAGPipeline(retriever=MockEmptyRetriever())
    resp = pipeline.answer_question("What is the battery chemistry of electric aircraft?")

    assert resp.is_evidence_sufficient is False
    assert "do not establish an answer" in resp.answer.lower()
    assert len(resp.claims) == 0
    assert resp.overall_confidence == "insufficient_evidence"
