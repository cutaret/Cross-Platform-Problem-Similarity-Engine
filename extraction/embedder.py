"""
Embedding provider for core_insight and composition_pattern vectors.

Supports two providers:
  - "local" (sentence-transformers, free, runs on CPU) — default
  - "voyage" (Voyage AI API, paid, higher quality)

Both produce L2-normalized vectors suitable for cosine similarity via dot product.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Unified embedding interface. Provider is selected from config.

    Usage:
        embedder = Embedder()
        vec = embedder.embed("This problem requires segment tree with lazy propagation")
        vecs = embedder.embed_batch(["text1", "text2", ...])
    """

    def __init__(self):
        settings = get_settings()
        self._provider = settings.embedding_provider
        self._dimension = settings.get_embedding_dimension()

        if self._provider == "local":
            self._impl = _LocalEmbedder(settings.local_embed_model)
        elif self._provider == "voyage":
            self._impl = _VoyageEmbedder(settings.voyage_api_key, settings.voyage_embed_model)
        else:
            raise ValueError(f"Unknown embedding provider: {self._provider}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a 1-D L2-normalized numpy array."""
        return self._impl.embed(text)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string (some providers use asymmetric retrieval)."""
        return self._impl.embed_query(text)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple texts. Returns an (N, D) L2-normalized numpy array."""
        return self._impl.embed_batch(texts)


# ── Local Embedder (sentence-transformers) ──────────────────────────────────

class _LocalEmbedder:
    """
    Uses sentence-transformers for free, local embeddings.
    Models are downloaded on first use (~30–130MB depending on model).

    Recommended models:
      - BAAI/bge-small-en-v1.5  (384-dim, 33MB, fast)
      - BAAI/bge-base-en-v1.5   (768-dim, 110MB, balanced)
      - sentence-transformers/all-MiniLM-L6-v2  (384-dim, 23MB, fastest)
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None  # lazy load

    def _get_model(self):
        if self._model is None:
            logger.info(f"Loading local embedding model: {self._model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"Model loaded (dimension={self._model.get_sentence_embedding_dimension()})")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Run:\n"
                    "  pip install sentence-transformers\n"
                    "Or switch to Voyage AI: set EMBEDDING_PROVIDER=voyage in .env"
                )
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE models benefit from a query prefix for retrieval
        if "bge" in self._model_name.lower():
            text = f"Represent this sentence for searching relevant passages: {text}"
        return self.embed(text)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
            batch_size=64,
        )
        return np.array(embeddings, dtype=np.float32)


# ── Voyage AI Embedder (paid API) ──────────────────────────────────────────

class _VoyageEmbedder:
    """
    Uses Voyage AI API for high-quality embeddings.
    Anthropic's recommended embedding partner.
    """

    _BATCH_SIZE = 128

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError(
                "VOYAGE_API_KEY not set. Get one at https://dash.voyageai.com/\n"
                "Or switch to local embeddings: set EMBEDDING_PROVIDER=local in .env"
            )
        import voyageai
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> np.ndarray:
        result = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="document",
        )
        vec = np.array(result.embeddings[0], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_query(self, text: str) -> np.ndarray:
        result = self._client.embed(
            texts=[text],
            model=self._model,
            input_type="query",
        )
        vec = np.array(result.embeddings[0], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            logger.info(
                f"Embedding batch {i // self._BATCH_SIZE + 1} "
                f"({len(batch)} texts, {i + len(batch)}/{len(texts)} total)"
            )
            result = self._client.embed(
                texts=list(batch),
                model=self._model,
                input_type="document",
            )
            all_embeddings.extend(result.embeddings)

        matrix = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return matrix / norms
