"""Retrieval evaluation metrics: Recall@K, MRR, and NDCG calculated from actual FAISS queries.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Set
import numpy as np

logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Calculate Recall@K: proportion of relevant items retrieved in top-k."""
    if not relevant_ids:
        return 1.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return min(1.0, hits / len(relevant_ids))


def compute_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)."""
    if not relevant_ids:
        return 1.0

    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 5) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not relevant_ids:
        return 1.0

    top_k = retrieved_ids[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k):
        rel = 1.0 if item in relevant_ids else 0.0
        dcg += rel / math.log2(idx + 2)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(min(k, len(relevant_ids))))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retrieval_results: List[Dict[str, Any]],
    k: int = 5
) -> Dict[str, Any]:
    """Calculate aggregated retrieval metrics over a set of query evaluation cases.

    Each case in retrieval_results must have:
      - 'retrieved_ids': List[str] (e.g. document titles or chunk IDs)
      - 'relevant_ids': List[str] (ground-truth relevant document titles or chunk IDs)
    """
    if not retrieval_results:
        return {"recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "total_queries": 0}

    recalls: List[float] = []
    mrrs: List[float] = []
    ndcgs: List[float] = []

    for case in retrieval_results:
        retrieved = case["retrieved_ids"]
        relevant = set(case["relevant_ids"])

        # Also support matching on lowercased or partial titles
        relevant_lower = {r.lower().strip() for r in relevant}
        matched_retrieved = []
        for r in retrieved:
            r_str = str(r).lower().strip()
            # Check if r matches any relevant title
            if any(rel in r_str or r_str in rel for rel in relevant_lower):
                matched_retrieved.append("MATCH")
            else:
                matched_retrieved.append(r)

        rel_target = {"MATCH"}

        rec = compute_recall_at_k(matched_retrieved, rel_target, k=k)
        mrr = compute_mrr(matched_retrieved, rel_target)
        ndcg = compute_ndcg_at_k(matched_retrieved, rel_target, k=k)

        recalls.append(rec)
        mrrs.append(mrr)
        ndcgs.append(ndcg)

    return {
        "k": k,
        "total_queries": len(retrieval_results),
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        "ndcg_at_k": round(float(np.mean(ndcgs)), 4)
    }
