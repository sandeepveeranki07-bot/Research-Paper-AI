"""Centralized configuration management for Research Intelligence Assistant.
Uses Pydantic for validation and python-dotenv for environment variable loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if present
load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseModel):
    # LLM Settings
    llm_provider: str = Field(default="openai", description="Provider: openai, anthropic, gemini, groq, ollama, custom")
    llm_api_key: Optional[str] = Field(default="", description="API key for the LLM provider")
    llm_model: str = Field(default="gpt-4o-mini", description="Model identifier")
    llm_base_url: Optional[str] = Field(default=None, description="Custom base URL for OpenAI-compatible or local API")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=8192)
    llm_timeout_seconds: float = Field(default=60.0, ge=5.0)

    # Embedding Settings
    embedding_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="SentenceTransformer embedding model name or local path"
    )
    embedding_device: str = Field(default="cpu", description="'cpu' or 'cuda'")
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_normalize: bool = Field(default=True, description="L2 normalize embeddings for cosine/inner-product similarity")

    # Ingestion & Chunking
    chunk_size_tokens: int = Field(default=700, ge=100, le=4000)
    chunk_overlap_tokens: int = Field(default=100, ge=0, le=1000)
    min_chunk_tokens: int = Field(default=50, ge=10)

    # Retrieval
    default_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_similarity_threshold: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Paths (Resolved relative to project base)
    data_dir: Path = Field(default=BASE_DIR / "data")
    papers_dir: Path = Field(default=BASE_DIR / "data" / "papers")
    index_dir: Path = Field(default=BASE_DIR / "data" / "indexes")
    evaluation_dir: Path = Field(default=BASE_DIR / "data" / "evaluation")
    sqlite_db_path: Path = Field(default=BASE_DIR / "data" / "research_assistant.db")
    faiss_index_path: Path = Field(default=BASE_DIR / "data" / "indexes" / "faiss_index.bin")
    faiss_metadata_path: Path = Field(default=BASE_DIR / "data" / "indexes" / "faiss_metadata.json")

    # OCR Fallback
    enable_ocr_fallback: bool = Field(default=True)
    tesseract_cmd: Optional[str] = Field(default=None)

    # API Server
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)

    def ensure_directories(self) -> None:
        """Ensure all required runtime data directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from environment variables."""
    data_dir_str = os.getenv("DATA_DIR")
    data_dir = Path(data_dir_str).resolve() if data_dir_str else (BASE_DIR / "data")

    papers_dir_str = os.getenv("PAPERS_DIR")
    papers_dir = Path(papers_dir_str).resolve() if papers_dir_str else (data_dir / "papers")

    index_dir_str = os.getenv("INDEX_DIR")
    index_dir = Path(index_dir_str).resolve() if index_dir_str else (data_dir / "indexes")

    evaluation_dir_str = os.getenv("EVALUATION_DIR")
    evaluation_dir = Path(evaluation_dir_str).resolve() if evaluation_dir_str else (data_dir / "evaluation")

    sqlite_db_str = os.getenv("SQLITE_DB_PATH")
    sqlite_db_path = Path(sqlite_db_str).resolve() if sqlite_db_str else (data_dir / "research_assistant.db")

    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
        llm_api_key=os.getenv("LLM_API_KEY", "") or None,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_base_url=os.getenv("LLM_BASE_URL", "") or None,
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "60.0")),

        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        embedding_normalize=os.getenv("EMBEDDING_NORMALIZE", "true").lower() in ("true", "1", "yes"),

        chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "700")),
        chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        min_chunk_tokens=int(os.getenv("MIN_CHUNK_TOKENS", "50")),

        default_top_k=int(os.getenv("DEFAULT_TOP_K", "5")),
        retrieval_similarity_threshold=float(os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.0")),

        data_dir=data_dir,
        papers_dir=papers_dir,
        index_dir=index_dir,
        evaluation_dir=evaluation_dir,
        sqlite_db_path=sqlite_db_path,
        faiss_index_path=index_dir / "faiss_index.bin",
        faiss_metadata_path=index_dir / "faiss_metadata.json",

        enable_ocr_fallback=os.getenv("ENABLE_OCR_FALLBACK", "true").lower() in ("true", "1", "yes"),
        tesseract_cmd=os.getenv("TESSERACT_CMD", "") or None,

        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
    )
    settings.ensure_directories()
    return settings


# Global settings singleton
settings: Settings = load_settings()


def get_settings() -> Settings:
    """Return the global configuration instance."""
    return settings
