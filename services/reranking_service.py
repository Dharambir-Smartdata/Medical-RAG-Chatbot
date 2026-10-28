"""
Reranking Service.
Uses a Cross-Encoder model to score (query, document chunk) relevance pairs,
selecting the top 5-8 most relevant chunks for LLM context injection.
"""

import logging
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    import torch
    from sentence_transformers import CrossEncoder
except ImportError:
    torch = None
    CrossEncoder = None


class RerankingService:
    """
    Cross-Encoder Reranker evaluating query-chunk relevance.
    """
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RerankingService, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        if self._model is None:
            self.model_name = model_name or getattr(
                settings, 'RERANKER_MODEL_NAME', 'cross-encoder/ms-marco-MiniLM-L-6-v2'
            )
            self.top_k = getattr(settings, 'RERANK_TOP_K', 8)
            self._load_model()

    def _load_model(self):
        """Loads CrossEncoder lazily with automatic repository name normalization."""
        if CrossEncoder is None:
            logger.warning("sentence_transformers.CrossEncoder is not installed. Using fallback reranking.")
            self._model = None
            return

        model_name = self.model_name
        # Normalize Xenova/ (ONNX-only repo) to cross-encoder/ (PyTorch weights)
        if model_name.startswith("Xenova/"):
            model_name = "cross-encoder/" + model_name[len("Xenova/"):]

        try:
            logger.info(f"Loading Cross-Encoder model: {model_name}")
            self._model = CrossEncoder(model_name)
        except Exception as e:
            logger.warning(f"Could not load Cross-Encoder '{model_name}': {e}. Trying default 'cross-encoder/ms-marco-MiniLM-L-6-v2'.")
            try:
                self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as inner_e:
                logger.error(f"Failed to load fallback Cross-Encoder: {inner_e}. Using retrieval score fallback.")
                self._model = None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reranks retrieved candidate chunks against the user question.
        Returns top_k sorted chunks with rerank_score attached.
        """
        if not candidates or not query:
            return []

        limit = top_k or self.top_k

        # If candidates are fewer than limit and no model, preserve current order
        if len(candidates) <= limit and self._model is None:
            return candidates

        if self._model is not None:
            try:
                pairs = [[query, c.get("chunk_text", "")] for c in candidates]
                scores = self._model.predict(pairs, batch_size=16, show_progress_bar=False)

                scored_candidates = []
                for idx, score in enumerate(scores):
                    item = dict(candidates[idx])
                    item["rerank_score"] = float(score)
                    scored_candidates.append(item)

                scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
                return scored_candidates[:limit]
            except Exception as e:
                logger.error(f"Cross-Encoder reranking error: {e}. Falling back to fusion scores.")

        # Fallback: sort by rrf_score or dense/sparse score
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("rrf_score", x.get("score", 0.0)),
            reverse=True
        )
        return sorted_candidates[:limit]
