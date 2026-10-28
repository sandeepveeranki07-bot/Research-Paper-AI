"""Pydantic schemas for citation-grounded RAG answers and claim attributions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CitationClaim(BaseModel):
    """An individual factual claim paired with verifiable source provenance."""
    claim_text: str = Field(..., description="The factual claim made in the answer.")
    source_document: str = Field(..., description="Original paper title or filename.")
    source_page: int = Field(..., description="Exact source page number.")
    source_chunk_id: Optional[str] = Field(default=None, description="Retrieved chunk identifier.")
    supporting_evidence: str = Field(..., description="Excerpt from the source passage directly supporting this claim.")
    evidence_relevance: str = Field(
        default="direct",
        description="Assessment of evidence relevance: 'direct', 'contextual', 'weak', 'unsupported'."
    )
    confidence: str = Field(
        default="high",
        description="Confidence or evidence strength: 'high', 'medium', 'low'."
    )
    is_valid: bool = Field(
        default=True,
        description="Whether this citation was validated against indexed documents and pages."
    )
    validation_status: str = Field(
        default="valid",
        description="'valid', 'invalid_document', 'invalid_page', 'unsupported_passage'."
    )
    validation_notes: Optional[str] = Field(default=None)

    @property
    def formatted_citation(self) -> str:
        return f"[{self.source_document}, p. {self.source_page}]"


class RAGResponse(BaseModel):
    """Structured response container for citation-grounded question answering."""
    query: str
    answer: str
    is_evidence_sufficient: bool = True
    claims: List[CitationClaim] = Field(default_factory=list)
    interpretations_and_inferences: List[str] = Field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations_rejected: int = 0
    overall_confidence: str = Field(
        default="high",
        description="'high', 'medium', 'low', 'insufficient_evidence'"
    )
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
