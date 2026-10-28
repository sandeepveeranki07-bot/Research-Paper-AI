"""High-performance embedding engine with ONNX Runtime and SentenceTransformers support.
Uses BAAI/bge-base-en-v1.5 (768 dimensions) with L2 normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages embedding generation using ONNX Runtime or SentenceTransformers with L2 normalization."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: Optional[bool] = None
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self.normalize = settings.embedding_normalize if normalize is None else normalize

        self._backend: str = "onnx"  # 'onnx' or 'sentence_transformers'
        self._onnx_session = None
        self._tokenizer = None
        self._st_model = None
        self._dimension: int = 768

    def _init_model(self) -> None:
        """Initialize the embedding backend (preferring ONNX runtime for robust cross-platform execution)."""
        if self._onnx_session is not None or self._st_model is not None:
            return

        # Attempt 1: ONNX Runtime with HuggingFace Tokenizers
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            from huggingface_hub import hf_hub_download

            repo_id = "Xenova/bge-base-en-v1.5"
            logger.info(f"Loading ONNX embedding model from {repo_id}...")

            onnx_path = hf_hub_download(repo_id=repo_id, filename="onnx/model_quantized.onnx")
            tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")

            self._tokenizer = Tokenizer.from_file(tok_path)
            self._tokenizer.enable_padding(length=512)
            self._tokenizer.enable_truncation(max_length=512)

            session_opts = ort.SessionOptions()
            session_opts.intra_op_num_threads = 4
            self._onnx_session = ort.InferenceSession(onnx_path, session_opts)
            self._backend = "onnx"
            self._dimension = 768
            logger.info("ONNX Embedding Engine initialized successfully (dim=768).")
            return
        except Exception as e_onnx:
            logger.warning(f"ONNX initialization failed or not available: {e_onnx}. Falling back to sentence-transformers.")

        # Attempt 2: SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            dummy = self._st_model.encode(["test"], show_progress_bar=False)
            self._dimension = dummy.shape[1]
            self._backend = "sentence_transformers"
            logger.info(f"SentenceTransformers loaded with dimension {self._dimension}.")
            return
        except Exception as e_st:
            logger.error(f"SentenceTransformers failed: {e_st}.")
            raise RuntimeError(f"Unable to load any embedding backend: {e_onnx} | {e_st}")

    def get_dimension(self) -> int:
        self._init_model()
        return self._dimension

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate float32 embeddings for a batch of texts."""
        if not texts:
            dim = self.get_dimension()
            return np.empty((0, dim), dtype=np.float32)

        self._init_model()

        if self._backend == "onnx":
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Preprocess batch
                encodings = [self._tokenizer.encode(t) for t in batch]
                input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
                attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
                token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }
                outputs = self._onnx_session.run(None, inputs)
                # outputs[0] is token_embeddings: shape (batch_size, seq_len, 768)
                # BGE uses CLS token embedding (index 0)
                cls_embeddings = outputs[0][:, 0, :]

                if self.normalize:
                    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    cls_embeddings = cls_embeddings / norms

                all_embeddings.append(cls_embeddings)

            return np.vstack(all_embeddings).astype(np.float32)

        else:
            embeddings = self._st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate float32 embedding for a search query (adding BGE instruction if applicable)."""
        processed_query = query.strip()
        if "bge" in self.model_name.lower() and not processed_query.startswith("Represent this sentence"):
            processed_query = f"Represent this sentence for searching relevant passages: {processed_query}"

        embeddings = self.embed_texts([processed_query], batch_size=1)
        return embeddings[0]
