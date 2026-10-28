"""Citation evaluation metrics: Citation Accuracy, Citation Coverage, and Error Breakdown.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


def compute_citation_metrics(evaluation_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate citation accuracy, coverage, and error distributions from evaluation records.

    Each record contains:
      - 'question_id': str
      - 'question': str
      - 'claims': List[Dict] (with 'is_valid', 'validation_status', 'claim_text', etc.)
      - 'has_citations': bool
    """
    total_questions = len(evaluation_records)
    if total_questions == 0:
        return {
            "total_questions": 0,
            "total_citations": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "unsupported_citations": 0,
            "citation_accuracy": 0.0,
            "citation_coverage": 0.0,
            "error_categories": {},
            "failure_cases": []
        }

    total_citations = 0
    valid_citations = 0
    invalid_doc_citations = 0
    invalid_page_citations = 0
    unsupported_citations = 0

    questions_with_citations = 0
    failure_cases: List[Dict[str, Any]] = []

    for rec in evaluation_records:
        claims = rec.get("claims", [])
        if claims:
            questions_with_citations += 1

        for c in claims:
            total_citations += 1
            status = c.get("validation_status", "valid")
            if c.get("is_valid", False) and status == "valid":
                valid_citations += 1
            else:
                if status == "invalid_document":
                    invalid_doc_citations += 1
                elif status == "invalid_page":
                    invalid_page_citations += 1
                elif status == "unsupported_passage":
                    unsupported_citations += 1

                # Log failure case
                if len(failure_cases) < 10:
                    failure_cases.append({
                        "question_id": rec.get("question_id"),
                        "question": rec.get("question"),
                        "claim_text": c.get("claim_text"),
                        "source_document": c.get("source_document"),
                        "source_page": c.get("source_page"),
                        "error_category": status,
                        "notes": c.get("validation_notes")
                    })

    # Total invalid citations
    invalid_citations = total_citations - valid_citations

    # Citation Accuracy: valid / total
    citation_accuracy = (valid_citations / total_citations) if total_citations > 0 else 0.0

    # Citation Coverage: questions with at least one citation / total questions
    citation_coverage = (questions_with_citations / total_questions) if total_questions > 0 else 0.0

    error_categories = {
        "invalid_document": invalid_doc_citations,
        "invalid_page": invalid_page_citations,
        "unsupported_passage": unsupported_citations
    }

    return {
        "total_questions": total_questions,
        "questions_with_citations": questions_with_citations,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "unsupported_citations": unsupported_citations,
        "citation_accuracy": round(citation_accuracy, 4),
        "citation_coverage": round(citation_coverage, 4),
        "error_categories": error_categories,
        "failure_cases": failure_cases
    }
