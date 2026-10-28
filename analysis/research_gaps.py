"""Extraction and synthesis of research gaps and open challenges across papers.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ResearchGapsAnalyzer:
    """Identifies unanswered questions, limitations, and future directions across papers."""

    def __init__(self) -> None:
        pass

    def extract_gaps(self, paper_records: List[Dict[str, Any]]) -> List[str]:
        """Synthesize research gaps from documented paper limitations and unresolved findings."""
        gaps: List[str] = []

        for p in paper_records:
            title = p.get("paper_title", "Document")
            limitations = p.get("limitations", "").strip()
            pages = p.get("source_pages", [1])
            page_cite = f"p. {pages[0]}" if pages else "p. 1"

            if limitations and limitations.lower() not in ("none", "n/a", "not specified"):
                gaps.append(f"[{title}, {page_cite}] Unaddressed limitation: {limitations}")

        # Check for cross-cutting themes
        if len(paper_records) >= 3:
            datasets = [p.get("dataset_or_setting", "") for p in paper_records if p.get("dataset_or_setting")]
            if len(set(datasets)) > 1:
                gaps.append(
                    "Benchmark fragmentation: The evaluated papers utilize disparate datasets and evaluation protocols, hindering direct cross-study meta-analysis."
                )

        if not gaps:
            gaps.append("The current evidence base does not explicitly document critical research gaps or limitations.")

        return gaps
