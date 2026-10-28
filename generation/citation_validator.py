"""Rigorous citation validation against indexed documents, page limits, and retrieved text.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from generation.answer_schema import CitationClaim
from retrieval.retriever import RetrievedChunk
from storage.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class CitationValidator:
    """Validates citations against SQLite document records and retrieved evidence chunks."""

    def __init__(self, doc_repo: Optional[DocumentRepository] = None) -> None:
        self.doc_repo = doc_repo or DocumentRepository()

    def validate_claims(
        self,
        claims: List[CitationClaim],
        retrieved_chunks: List[RetrievedChunk]
    ) -> List[CitationClaim]:
        """Validate every citation claim against retrieved evidence and repository metadata."""
        if not claims:
            return []

        # Build quick lookup from retrieved chunks
        chunk_map = {c.chunk_id: c for c in retrieved_chunks}

        # Cache known documents
        all_docs = self.doc_repo.list_documents()
        doc_by_id = {d["id"]: d for d in all_docs}
        doc_by_title_lower = {d["title"].lower().strip(): d for d in all_docs if d.get("title")}
        doc_by_filename_lower = {d["filename"].lower().strip(): d for d in all_docs}

        validated_claims: List[CitationClaim] = []

        for claim in claims:
            cited_doc_str = claim.source_document.strip()
            cited_page = claim.source_page

            # 1. Resolve cited document
            matched_doc = None
            doc_lower = cited_doc_str.lower()

            if doc_lower in doc_by_title_lower:
                matched_doc = doc_by_title_lower[doc_lower]
            elif doc_lower in doc_by_filename_lower:
                matched_doc = doc_by_filename_lower[doc_lower]
            else:
                # Substring match on title or filename
                for title, d in doc_by_title_lower.items():
                    if doc_lower in title or title in doc_lower:
                        matched_doc = d
                        break
                if not matched_doc:
                    for fname, d in doc_by_filename_lower.items():
                        if doc_lower in fname or fname in doc_lower:
                            matched_doc = d
                            break

            if not matched_doc:
                # Document does not exist in the collection
                claim.is_valid = False
                claim.validation_status = "invalid_document"
                claim.validation_notes = f"Referenced document '{cited_doc_str}' does not exist in index."
                validated_claims.append(claim)
                continue

            # 2. Validate cited page number against total pages
            total_pages = matched_doc.get("total_pages", 0)
            if cited_page < 1 or (total_pages > 0 and cited_page > total_pages):
                claim.is_valid = False
                claim.validation_status = "invalid_page"
                claim.validation_notes = f"Cited page {cited_page} is out of bounds (Document has {total_pages} pages)."
                validated_claims.append(claim)
                continue

            # 3. Check supporting passage against retrieved chunks
            # Find candidate chunks for this document and page
            matching_chunks = [
                c for c in retrieved_chunks
                if (c.document_id == matched_doc["id"] or c.document_filename.lower() == matched_doc["filename"].lower())
                and c.start_page <= cited_page <= c.end_page
            ]

            if not matching_chunks:
                # Try finding any chunk for this document in retrieved chunks
                matching_chunks = [
                    c for c in retrieved_chunks
                    if c.document_id == matched_doc["id"] or c.document_filename.lower() == matched_doc["filename"].lower()
                ]

            if not matching_chunks:
                claim.is_valid = False
                claim.validation_status = "unsupported_passage"
                claim.validation_notes = f"No retrieved chunks from '{matched_doc['title']}' on page {cited_page} support this claim."
                validated_claims.append(claim)
                continue

            # Verify that the supporting evidence is grounded in the chunk text
            evidence = claim.supporting_evidence.strip().lower()
            combined_chunk_text = " ".join(c.text.lower() for c in matching_chunks)

            # Check textual containment or significant overlap
            is_supported = self._verify_textual_support(evidence, combined_chunk_text, claim.claim_text.lower())

            if is_supported:
                claim.is_valid = True
                claim.validation_status = "valid"
                # Standardize source_document to the canonical title
                claim.source_document = matched_doc.get("title") or matched_doc["filename"]
                if not claim.source_chunk_id and matching_chunks:
                    claim.source_chunk_id = matching_chunks[0].chunk_id
            else:
                claim.is_valid = False
                claim.validation_status = "unsupported_passage"
                claim.validation_notes = "Evidence text is not substantiated by the retrieved source chunks."

            validated_claims.append(claim)

        return validated_claims

    def _verify_textual_support(self, evidence: str, chunk_text: str, claim: str) -> bool:
        """Verify whether the evidence quote or claim words are substantiated by chunk text."""
        if not evidence and not claim:
            return False

        # If evidence quote is directly contained
        if len(evidence) > 15 and evidence in chunk_text:
            return True

        # Check token overlap between evidence words and chunk text
        evidence_words = set(re.findall(r"\w{4,}", evidence))
        if evidence_words:
            matched_words = sum(1 for w in evidence_words if w in chunk_text)
            overlap_ratio = matched_words / len(evidence_words)
            if overlap_ratio >= 0.65:
                return True

        # Check claim keywords
        claim_words = set(re.findall(r"\w{4,}", claim))
        if claim_words:
            matched_claim_words = sum(1 for w in claim_words if w in chunk_text)
            claim_ratio = matched_claim_words / len(claim_words)
            if claim_ratio >= 0.70:
                return True

        return False
