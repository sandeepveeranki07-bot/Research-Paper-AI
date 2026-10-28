"""Unit tests for evaluation dataset integrity and metric calculations.
"""

from __future__ import annotations

import pytest

from evaluation.citation_metrics import compute_citation_metrics
from evaluation.evaluation_dataset import get_curated_evaluation_dataset
from evaluation.reports import generate_markdown_report


def test_evaluation_dataset_completeness() -> None:
    dataset = get_curated_evaluation_dataset()
    # User prompt requirement: at least 100 test questions
    assert len(dataset) >= 100

    categories = {q.category for q in dataset}
    assert "Architectures" in categories
    assert "RAG & Retrieval" in categories
    assert "Activity Recognition" in categories
    assert "Optimization" in categories
    assert "Multi-Paper Analysis" in categories

    for q in dataset:
        assert q.question_id.startswith("Q")
        assert len(q.question) > 15
        assert len(q.expected_document_titles) > 0
        assert len(q.expected_pages) > 0
        assert len(q.reference_answer) > 20


def test_citation_metrics_calculation() -> None:
    records = [
        {
            "question_id": "Q001",
            "question": "What is self-attention?",
            "claims": [
                {"claim_text": "Claim 1", "is_valid": True, "validation_status": "valid"},
                {"claim_text": "Claim 2", "is_valid": True, "validation_status": "valid"}
            ]
        },
        {
            "question_id": "Q002",
            "question": "What is ViT?",
            "claims": [
                {"claim_text": "Claim 3", "is_valid": False, "validation_status": "invalid_page", "validation_notes": "Out of bounds"},
                {"claim_text": "Claim 4", "is_valid": True, "validation_status": "valid"}
            ]
        }
    ]

    metrics = compute_citation_metrics(records)
    assert metrics["total_questions"] == 2
    assert metrics["total_citations"] == 4
    assert metrics["valid_citations"] == 3
    assert metrics["invalid_citations"] == 1
    # 3 valid out of 4 total = 0.75
    assert metrics["citation_accuracy"] == 0.75
    assert metrics["citation_coverage"] == 1.0
    assert metrics["error_categories"]["invalid_page"] == 1
    assert len(metrics["failure_cases"]) == 1


def test_markdown_report_generation() -> None:
    retrieval_stats = {"recall_at_k": 0.88, "mrr": 0.92, "ndcg_at_k": 0.90}
    citation_stats = {
        "total_questions": 105,
        "total_citations": 210,
        "valid_citations": 195,
        "invalid_citations": 15,
        "unsupported_citations": 5,
        "citation_accuracy": 0.9286,
        "citation_coverage": 1.0,
        "error_categories": {"invalid_page": 10, "unsupported_passage": 5},
        "failure_cases": []
    }
    latency_stats = {"avg_retrieval_latency_ms": 12.5, "avg_generation_latency_ms": 45.2}
    ingestion_stats = {"total_documents": 8, "total_pages": 45, "total_chunks": 120, "ingestion_failures": 0}

    report = generate_markdown_report(retrieval_stats, citation_stats, latency_stats, ingestion_stats)

    assert "# Research Intelligence Assistant — Evaluation Report" in report
    assert "88.00%" in report
    assert "92.86%" in report
    assert "Average Semantic Retrieval Latency" in report
