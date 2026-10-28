"""Unit tests for multi-paper comparison and contradiction detection.
"""

from __future__ import annotations

import pytest

from analysis.contradiction_detection import ContradictionDetector
from analysis.paper_comparison import PaperComparisonRow
from analysis.research_gaps import ResearchGapsAnalyzer


def test_contradiction_detector_direct_conflict() -> None:
    detector = ContradictionDetector()

    p1 = {
        "paper_title": "Paper A",
        "main_finding": "Proposed method outperforms all recurrent baselines by 5 BLEU.",
        "methodology": "Pure self-attention",
        "dataset_or_setting": "WMT English-German",
        "source_pages": [1]
    }
    p2 = {
        "paper_title": "Paper B",
        "main_finding": "Recurrent neural architectures underperform pure attention models on small datasets.",
        "methodology": "Recurrent LSTM",
        "dataset_or_setting": "WMT English-German",
        "source_pages": [2]
    }

    contradictions = detector.analyze_contradictions([p1, p2])
    assert len(contradictions) >= 1
    c = contradictions[0]
    assert "Paper A" in c.citation_a
    assert "Paper B" in c.citation_b
    assert c.disagreement_type in ("direct_contradiction", "differing_results", "methodological_divergence")


def test_contradiction_detector_methodological_divergence() -> None:
    detector = ContradictionDetector()

    p1 = {
        "paper_title": "Paper X",
        "main_finding": "Achieves higher accuracy on ImageNet-1K.",
        "dataset_or_setting": "ImageNet-1K (1.3M images)",
        "source_pages": [3]
    }
    p2 = {
        "paper_title": "Paper Y",
        "main_finding": "Requires pre-training on JFT-300M to beat CNN baselines.",
        "dataset_or_setting": "JFT-300M (300M images)",
        "source_pages": [4]
    }

    contradictions = detector.analyze_contradictions([p1, p2])
    assert len(contradictions) >= 1
    assert "Divergence may stem from benchmark variability" in contradictions[0].methodological_differences


def test_research_gaps_extraction() -> None:
    analyzer = ResearchGapsAnalyzer()

    records = [
        {
            "paper_title": "Paper Alpha",
            "limitations": "High computational cost during inference on long sequences.",
            "source_pages": [5]
        },
        {
            "paper_title": "Paper Beta",
            "limitations": "Requires full 16-bit precision weights in GPU VRAM.",
            "source_pages": [3]
        }
    ]

    gaps = analyzer.extract_gaps(records)
    assert len(gaps) >= 2
    assert any("computational cost" in g.lower() for g in gaps)
    assert any("16-bit precision" in g.lower() for g in gaps)
