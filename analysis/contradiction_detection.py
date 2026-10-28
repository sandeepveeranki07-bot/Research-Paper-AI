"""Cross-paper contradiction detection and disagreement analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ContradictionItem(BaseModel):
    """Structured representation of a cross-paper disagreement or contradiction."""
    topic: str = Field(..., description="Topic or research question under debate.")
    claim_a: str = Field(..., description="First claim.")
    citation_a: str = Field(..., description="Supporting citation for Claim A e.g. [Paper A, p. 5].")
    claim_b: str = Field(..., description="Second conflicting or divergent claim.")
    citation_b: str = Field(..., description="Supporting citation for Claim B e.g. [Paper B, p. 8].")
    disagreement_type: str = Field(
        default="direct_contradiction",
        description="'direct_contradiction', 'differing_results', 'methodological_divergence', 'partial_agreement'"
    )
    methodological_differences: str = Field(
        default="",
        description="Explaining differences in datasets, assumptions, or measurement metrics."
    )
    evidence_strength: str = Field(
        default="high",
        description="'high', 'medium', 'low'"
    )
    is_explicit_contradiction: bool = Field(
        default=False,
        description="True if explicit disagreement; False if inferred from differing results."
    )
    unresolved_aspects: Optional[str] = Field(default=None)


class ContradictionDetector:
    """Detects and categorizes contradictory claims between pairs of papers."""

    def __init__(self) -> None:
        pass

    def analyze_contradictions(
        self,
        paper_findings: List[Dict[str, Any]]
    ) -> List[ContradictionItem]:
        """Analyze pairs of paper findings to detect potential contradictions or divergencies."""
        contradictions: List[ContradictionItem] = []
        n = len(paper_findings)
        if n < 2:
            return []

        for i in range(n):
            for j in range(i + 1, n):
                p1 = paper_findings[i]
                p2 = paper_findings[j]

                # Check if both address comparable topics or findings
                item = self._compare_paper_pair(p1, p2)
                if item:
                    contradictions.append(item)

        return contradictions

    def _compare_paper_pair(
        self,
        p1: Dict[str, Any],
        p2: Dict[str, Any]
    ) -> Optional[ContradictionItem]:
        """Examine whether two papers report conflicting findings or methodological divergencies."""
        f1 = p1.get("main_finding", "").lower()
        f2 = p2.get("main_finding", "").lower()

        t1 = p1.get("paper_title", "Paper 1")
        t2 = p2.get("paper_title", "Paper 2")

        pages1 = p1.get("source_pages", [1])
        pages2 = p2.get("source_pages", [1])
        cite1 = f"[{t1}, p. {pages1[0] if pages1 else 1}]"
        cite2 = f"[{t2}, p. {pages2[0] if pages2 else 1}]"

        m1 = p1.get("methodology", "N/A")
        m2 = p2.get("methodology", "N/A")
        d1 = p1.get("dataset_or_setting", "N/A")
        d2 = p2.get("dataset_or_setting", "N/A")

        # Conflict heuristic keywords
        conflict_indicators = [
            ("outperform", "underperform"),
            ("superior", "inferior"),
            ("effective", "ineffective"),
            ("increases", "decreases"),
            ("improves", "degrades"),
            ("robust", "vulnerable"),
            ("necessary", "redundant"),
            ("scalable", "unscalable")
        ]

        has_conflict = False
        disagreement_type = "differing_results"
        is_explicit = False

        for pos, neg in conflict_indicators:
            if (pos in f1 and neg in f2) or (neg in f1 and pos in f2):
                has_conflict = True
                disagreement_type = "direct_contradiction"
                is_explicit = True
                break

        # Check if settings differ
        if d1.lower() != d2.lower() and d1 != "N/A" and d2 != "N/A":
            diff_note = f"Paper 1 evaluated on '{d1}', whereas Paper 2 evaluated on '{d2}'. Divergence may stem from benchmark variability rather than fundamental theoretical incompatibility."
            if not has_conflict and ("higher" in f1 or "higher" in f2 or "better" in f1 or "better" in f2):
                has_conflict = True
                disagreement_type = "methodological_divergence"
        else:
            diff_note = "Evaluated on identical or comparable experimental settings."

        if has_conflict:
            return ContradictionItem(
                topic=f"Comparative performance / findings between {t1} and {t2}",
                claim_a=p1.get("main_finding", ""),
                citation_a=cite1,
                claim_b=p2.get("main_finding", ""),
                citation_b=cite2,
                disagreement_type=disagreement_type,
                methodological_differences=diff_note,
                evidence_strength="high" if is_explicit else "medium",
                is_explicit_contradiction=is_explicit,
                unresolved_aspects="Further controlled ablation under standardized benchmark conditions is required."
            )

        return None
