"""Unit and integration tests for embeddings, FAISS indexing, filtering, and retrieval metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.retrieval_metrics import compute_mrr, compute_ndcg_at_k, compute_recall_at_k, evaluate_retrieval
from retrieval.embeddings import EmbeddingEngine
from retrieval.faiss_index import FAISSIndexManager
from retrieval.filters import MetadataFilter
from storage.index_repository import IndexRepository


def test_embedding_engine_dimension_and_norm() -> None:
    engine = EmbeddingEngine()
    dim = engine.get_dimension()
    assert dim == 768

    embeddings = engine.embed_texts(["Transformer architectures in deep learning.", "Reinforcement learning."])
    assert embeddings.shape == (2, 768)
    assert embeddings.dtype == np.float32

    # Verify L2 normalization: norm should equal ~1.0
    for vec in embeddings:
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4


def test_faiss_index_manager_crud(tmp_path) -> None:
    index_file = tmp_path / "test.index"
    meta_file = tmp_path / "test.json"
    repo = IndexRepository(index_path=index_file, metadata_path=meta_file)

    mgr = FAISSIndexManager(dimension=4, index_repo=repo)
    assert mgr.total_vectors() == 0

    # Add 3 normalized vectors
    v1 = np.array([1, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 1, 0, 0], dtype=np.float32)
    v3 = np.array([0, 0, 1, 0], dtype=np.float32)
    vectors = np.vstack([v1, v2, v3])
    ids = [101, 102, 103]

    mgr.add_vectors(vectors, ids)
    assert mgr.total_vectors() == 3

    # Search query matching v1 exactly
    query = np.array([1, 0, 0, 0], dtype=np.float32)
    scores, ret_ids = mgr.search(query, k=2)

    assert len(ret_ids) == 2
    assert ret_ids[0] == 101
    assert abs(scores[0] - 1.0) < 1e-5

    # Remove vector 101
    mgr.remove_ids([101])
    assert mgr.total_vectors() == 2

    # Save and reload
    mgr.save()
    reloaded_mgr = FAISSIndexManager(dimension=4, index_repo=repo)
    assert reloaded_mgr.total_vectors() == 2


def test_metadata_filters() -> None:
    chunk = {
        "chunk_id": "c1",
        "document_id": "doc_abc",
        "document_filename": "paper1.pdf",
        "section_heading": "3. Methodology"
    }
    doc_meta = {
        "publication_year": 2021,
        "authors": ["Alice Smith", "Bob Jones"]
    }

    # Matching filter
    f_pass = MetadataFilter(document_ids=["doc_abc"], min_year=2020, max_year=2022)
    assert f_pass.matches(chunk, doc_meta) is True

    # Year mismatch
    f_fail_year = MetadataFilter(min_year=2023)
    assert f_fail_year.matches(chunk, doc_meta) is False

    # Author mismatch
    f_fail_author = MetadataFilter(authors=["Charlie"])
    assert f_fail_author.matches(chunk, doc_meta) is False


def test_retrieval_metrics_calculation() -> None:
    # 1. Recall@K
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevant = {"doc_c", "doc_x"}
    # doc_c is retrieved within top 5 -> 1 hit out of 2 relevant
    rec = compute_recall_at_k(retrieved, relevant, k=5)
    assert rec == 0.5

    # 2. MRR (First relevant is doc_c at rank 3)
    mrr = compute_mrr(retrieved, relevant)
    assert round(mrr, 4) == round(1.0 / 3.0, 4)

    # 3. NDCG@K
    ndcg = compute_ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 < ndcg <= 1.0

    # 4. Batch evaluation
    cases = [
        {"retrieved_ids": ["Attention Is All You Need", "ViT"], "relevant_ids": ["Attention Is All You Need"]},
        {"retrieved_ids": ["ResNet", "ViT"], "relevant_ids": ["ViT"]}
    ]
    batch_res = evaluate_retrieval(cases, k=5)
    assert batch_res["recall_at_k"] == 1.0
    assert batch_res["total_queries"] == 2
