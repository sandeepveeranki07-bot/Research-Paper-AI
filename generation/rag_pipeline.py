"""RAG Answer Generation Pipeline with strict citation verification and latency instrumentation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from config import get_settings
from generation.answer_schema import CitationClaim, RAGResponse
from generation.citation_validator import CitationValidator
from generation.llm_provider import BaseLLMProvider, get_llm_provider
from generation.prompts import RAG_SYSTEM_PROMPT, format_retrieval_context
from retrieval.filters import MetadataFilter
from retrieval.retriever import RetrievedChunk, SemanticRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline coordinating retrieval, synthesis, and citation verification."""

    def __init__(
        self,
        retriever: Optional[SemanticRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        validator: Optional[CitationValidator] = None
    ) -> None:
        self.retriever = retriever or SemanticRetriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.validator = validator or CitationValidator()

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[MetadataFilter] = None,
        min_score: Optional[float] = None
    ) -> RAGResponse:
        """Execute full RAG generation with verifiable, page-level citations."""
        # 1. Retrieval & Latency Tracking
        t0 = time.perf_counter()
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            min_score=min_score
        )
        t1 = time.perf_counter()
        retrieval_latency_ms = (t1 - t0) * 1000.0

        # If no chunks were retrieved or evidence is missing
        if not retrieved_chunks:
            return RAGResponse(
                query=query,
                answer="The available papers do not establish an answer to this question because no relevant evidence was retrieved from the indexed collection.",
                is_evidence_sufficient=False,
                claims=[],
                interpretations_and_inferences=[],
                retrieved_evidence=[],
                total_citations=0,
                valid_citations=0,
                invalid_citations_rejected=0,
                overall_confidence="insufficient_evidence",
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                generation_latency_ms=0.0
            )

        # 2. Context Construction
        context_str = format_retrieval_context(retrieved_chunks)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"USER QUESTION:\n{query}\n\nRETRIEVED RESEARCH EVIDENCE:\n{context_str}\n\nPlease generate a citation-grounded response as JSON."
            }
        ]

        # 3. LLM Generation
        t2 = time.perf_counter()
        raw_output = self.llm_provider.generate_completion(messages, json_mode=True)
        t3 = time.perf_counter()
        generation_latency_ms = (t3 - t2) * 1000.0

        # 4. Parse JSON Response
        parsed = self._safe_parse_json(raw_output)

        # 5. Extract and Validate Claims
        raw_claims_data = parsed.get("claims", [])
        candidate_claims: List[CitationClaim] = []
        for c in raw_claims_data:
            try:
                candidate_claims.append(CitationClaim(**c))
            except Exception as e:
                logger.warning(f"Failed to parse claim: {c}, error: {e}")

        # Validate each citation against SQLite documents and page ranges
        validated_claims = self.validator.validate_claims(candidate_claims, retrieved_chunks)

        # Filter out rejected/invalid citations from presentation or mark them
        valid_claims = [c for c in validated_claims if c.is_valid]
        invalid_count = len(validated_claims) - len(valid_claims)

        is_sufficient = parsed.get("is_evidence_sufficient", True)
        answer_text = parsed.get("answer", "")

        # If evidence was deemed insufficient or no valid claims exist
        if not is_sufficient or (not valid_claims and "do not establish" in answer_text.lower()):
            overall_conf = "insufficient_evidence"
            is_sufficient = False
        else:
            overall_conf = parsed.get("overall_confidence", "high")

        return RAGResponse(
            query=query,
            answer=answer_text,
            is_evidence_sufficient=is_sufficient,
            claims=valid_claims,
            interpretations_and_inferences=parsed.get("interpretations_and_inferences", []),
            retrieved_evidence=[c.to_dict() for c in retrieved_chunks],
            total_citations=len(validated_claims),
            valid_citations=len(valid_claims),
            invalid_citations_rejected=invalid_count,
            overall_confidence=overall_conf,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2)
        )

    def _safe_parse_json(self, raw_text: str) -> Dict[str, Any]:
        """Robust JSON extraction from LLM response text."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            # Fallback regex search for JSON object
            import re
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            logger.error(f"Failed to parse JSON response: {raw_text[:200]}")
            return {
                "answer": raw_text,
                "is_evidence_sufficient": True,
                "claims": [],
                "interpretations_and_inferences": [],
                "overall_confidence": "medium"
            }
