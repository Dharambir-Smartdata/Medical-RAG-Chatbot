"""
Hybrid Retrieval Service.
Combines Dense Semantic Vector Search and Sparse BM25 Keyword Search
using Reciprocal Rank Fusion (RRF) and deduplication.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from django.conf import settings

from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService
from services.sparse_retrieval_service import SparseRetrievalService

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """
    Hybrid RAG coordinator fusing Dense (Qdrant) and Sparse (BM25) retrieval results.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        qdrant_service: Optional[QdrantService] = None,
        sparse_service: Optional[SparseRetrievalService] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.qdrant_service = qdrant_service or QdrantService()
        self.sparse_service = sparse_service or SparseRetrievalService()

        self.dense_top_k = getattr(settings, 'DENSE_TOP_K', 20)
        self.sparse_top_k = getattr(settings, 'SPARSE_TOP_K', 20)
        self.rrf_k = getattr(settings, 'RRF_K', 60)

    @classmethod
    def reciprocal_rank_fusion(
        cls,
        ranked_lists: List[List[Dict[str, Any]]],
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Calculates RRF score:
        RRF(d) = sum(1.0 / (rrf_k + rank(d)))
        Deduplicates chunks by chunk_id and aggregates scores.
        """
        rrf_scores = defaultdict(float)
        chunks_map: Dict[str, Dict[str, Any]] = {}
        retrieval_sources = defaultdict(set)

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, start=1):
                cid = item.get("chunk_id")
                if not cid:
                    continue

                # RRF calculation
                rrf_scores[cid] += 1.0 / (rrf_k + rank)

                if cid not in chunks_map:
                    chunks_map[cid] = dict(item)

                source = item.get("retrieval_source", "unknown")
                retrieval_sources[cid].add(source)

        # Build fused result list
        fused_results = []
        for cid, score in rrf_scores.items():
            record = chunks_map[cid]
            record["rrf_score"] = score
            record["retrieval_sources"] = list(retrieval_sources[cid])
            fused_results.append(record)

        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results

    def retrieve(
        self,
        query: str,
        corpus_chunks: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None,
        dense_limit: Optional[int] = None,
        sparse_limit: Optional[int] = None,
        fusion_limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Executes dense and sparse retrieval, applies RRF fusion and deduplication.
        """
        dense_k = dense_limit or self.dense_top_k
        sparse_k = sparse_limit or self.sparse_top_k

        # 1. Dense Semantic Retrieval
        query_vector = self.embedding_service.embed_query(query)
        dense_results = self.qdrant_service.search_dense(
            query_vector=query_vector,
            limit=dense_k,
            document_id=document_id
        )

        # 2. Sparse BM25 Keyword Retrieval
        sparse_results = []
        if corpus_chunks:
            sparse_results = self.sparse_service.search_sparse(
                query=query,
                corpus_chunks=corpus_chunks,
                limit=sparse_k,
                document_id=document_id
            )

        # If both empty, return empty
        if not dense_results and not sparse_results:
            return []

        # If only one source is available
        if not sparse_results:
            return dense_results[:fusion_limit]
        if not dense_results:
            return sparse_results[:fusion_limit]

        # 3. Reciprocal Rank Fusion
        fused = self.reciprocal_rank_fusion(
            ranked_lists=[dense_results, sparse_results],
            rrf_k=self.rrf_k
        )

        return fused[:fusion_limit]
