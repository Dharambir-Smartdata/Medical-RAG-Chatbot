"""
Embedding Module.
Uses Sentence Transformers to generate dense embeddings for document chunks and queries.
Dynamic dimension detection, batch embedding, and normalization.
"""

import logging
from typing import List, Union
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    torch = None
    SentenceTransformer = None


class EmbeddingService:
    """
    Singleton-capable embedding service using Sentence Transformers.
    """
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._model is None:
            self.model_name = model_name or getattr(
                settings, 'EMBEDDING_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'
            )
            self._dimension = None
            self._load_model()

    def _load_model(self):
        """Loads the sentence transformer model lazily."""
        if SentenceTransformer is None:
            logger.warning("SentenceTransformer is not installed. Using fallback embedding generation.")
            self._model = None
            self._dimension = 384
            return

        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            if hasattr(self._model, 'get_embedding_dimension'):
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"Failed to load embedding model {self.model_name}: {e}. Trying fallback 'sentence-transformers/all-MiniLM-L6-v2'.")
            try:
                self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self._dimension = 384
            except Exception as inner_e:
                logger.error(f"Failed to load fallback embedding model: {inner_e}.")
                self._model = None
                self._dimension = 384

    @property
    def dimension(self) -> int:
        """Dynamically detected embedding dimension."""
        if self._dimension is None:
            if self._model is not None:
                if hasattr(self._model, 'get_embedding_dimension'):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
            else:
                self._dimension = 384
        return self._dimension

    def embed_texts(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> List[List[float]]:
        """Generates dense embeddings for a batch of text chunks."""
        if not texts:
            return []

        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embeddings.tolist()

        # Fallback pseudo-embedding generator for test/offline environments
        fallback_dim = self.dimension
        results = []
        for text in texts:
            np.random.seed(abs(hash(text)) % (2**32))
            vec = np.random.randn(fallback_dim).astype(np.float32)
            if normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            results.append(vec.tolist())
        return results

    def embed_query(self, query: str, normalize: bool = True) -> List[float]:
        """Generates embedding for a single user query."""
        if not query or not query.strip():
            return [0.0] * self.dimension

        if self._model is not None:
            embedding = self._model.encode(
                query,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            return embedding.tolist()

        return self.embed_texts([query], normalize=normalize)[0]
