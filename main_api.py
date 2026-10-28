"""FastAPI backend application for the Research Intelligence Assistant.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analysis.paper_comparison import ComparisonResult, PaperComparator
from config import get_settings
from evaluation.run_evaluation import run_full_evaluation
from generation.answer_schema import RAGResponse
from generation.rag_pipeline import RAGPipeline
from ingestion.pipeline import IngestionPipeline, IngestionResult
from retrieval.filters import MetadataFilter
from retrieval.retriever import RetrievedChunk, SemanticRetriever
from storage.database import get_db, init_database
from storage.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Research Intelligence Assistant API",
    description="Production RAG API for multi-paper indexing, retrieval, citation-grounded answering, and comparative analysis.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Startup event
@app.on_event("startup")
def on_startup() -> None:
    init_database()
    logger.info("Database and storage initialized.")


# Request models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=5, ge=1, le=50)
    document_ids: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None


class CompareRequest(BaseModel):
    query: Optional[str] = "Compare methodologies, datasets, findings, and limitations across papers."
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = 15


# Endpoints
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    doc_repo = DocumentRepository()
    stats = doc_repo.get_document_stats()
    return {
        "status": "healthy",
        "service": "Research Intelligence Assistant",
        "index_stats": stats
    }


@app.get("/api/documents")
def list_documents(search: Optional[str] = None) -> List[Dict[str, Any]]:
    doc_repo = DocumentRepository()
    return doc_repo.list_documents(search=search)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, Any]:
    pipeline = IngestionPipeline()
    success = pipeline.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": f"Document {doc_id} deleted successfully.", "deleted": True}


@app.post("/api/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> Dict[str, Any]:
    settings = get_settings()
    saved_paths: List[Path] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        dest = settings.papers_dir / file.filename
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(dest)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded.")

    pipeline = IngestionPipeline()
    res = pipeline.ingest_files(saved_paths)
    return {
        "summary": res.summary(),
        "successful_docs": res.successful_docs,
        "failed_docs": res.failed_docs,
        "skipped_duplicates": res.skipped_duplicates,
        "errors": res.errors
    }


@app.post("/api/retrieve")
def retrieve_evidence(req: QueryRequest) -> List[Dict[str, Any]]:
    retriever = SemanticRetriever()
    filters = MetadataFilter(
        document_ids=req.document_ids,
        authors=req.authors,
        min_year=req.min_year,
        max_year=req.max_year
    )
    chunks = retriever.retrieve(query=req.query, top_k=req.top_k, filters=filters)
    return [c.to_dict() for c in chunks]


@app.post("/api/chat", response_model=RAGResponse)
def answer_question(req: QueryRequest) -> RAGResponse:
    filters = MetadataFilter(
        document_ids=req.document_ids,
        authors=req.authors,
        min_year=req.min_year,
        max_year=req.max_year
    )
    pipeline = RAGPipeline()
    return pipeline.answer_question(query=req.query, top_k=req.top_k, filters=filters)


@app.post("/api/compare", response_model=ComparisonResult)
def compare_papers(req: CompareRequest) -> ComparisonResult:
    comparator = PaperComparator()
    return comparator.compare(query=req.query or "", document_ids=req.document_ids, top_k=req.top_k or 15)


@app.get("/api/evaluation/latest")
def get_latest_evaluation() -> Dict[str, Any]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM evaluation_runs ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {"message": "No evaluation runs recorded yet."}
        return dict(row)


@app.post("/api/evaluation/run")
def trigger_evaluation(max_questions: int = Query(default=105, ge=5, le=105)) -> Dict[str, Any]:
    results = run_full_evaluation(max_questions=max_questions)
    return {
        "message": "Evaluation completed successfully.",
        "results": results
    }
