"""
Qdrant Vector Database Service.
Handles vector collection management, chunk upserts, dense similarity search,
and document-level vector deletion.
Supports both remote Docker Qdrant and local file/in-memory fallback.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
    from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
except ImportError:
    QdrantClient = None
    rest_models = None


class QdrantService:
    """
    Service wrapper for interacting with Qdrant vector database.
    Reuses connection pool across requests to avoid network handshake latency.
    """
    _shared_client = None

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or getattr(settings, 'QDRANT_COLLECTION', 'medical_documents')
        if QdrantService._shared_client is None:
            QdrantService._shared_client = self._initialize_client()
        self.client = QdrantService._shared_client

    def _initialize_client(self):
        """Initializes client connecting to Docker Qdrant, falling back to local storage."""
        if QdrantClient is None:
            logger.warning("qdrant-client package is not installed. Running in mock vector mode.")
            return None

        qdrant_url = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')
        api_key = getattr(settings, 'QDRANT_API_KEY', None) or None
        local_path = getattr(settings, 'QDRANT_LOCAL_PATH', './qdrant_storage')

        # Try connecting to remote/Docker Qdrant first
        if qdrant_url:
            try:
                client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=4.0)
                client.get_collections()
                logger.info(f"Connected to Qdrant at {qdrant_url}")
                return client
            except Exception as e:
                logger.warning(
                    f"Remote Qdrant at {qdrant_url} is not accessible ({e}). "
                    f"Falling back to local disk storage at {local_path}."
                )

        # Fallback to local disk-based Qdrant client
        try:
            client = QdrantClient(path=local_path)
            logger.info(f"Initialized local disk Qdrant storage at {local_path}")
            return client
        except Exception as e:
            logger.warning(f"Failed to initialize local Qdrant at {local_path}: {e}. Falling back to in-memory.")
            return QdrantClient(":memory:")

    def ensure_collection(self, vector_size: int = 384):
        """Creates the collection if it does not already exist."""
        if self.client is None or rest_models is None:
            return

        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                logger.info(f"Creating Qdrant collection: {self.collection_name} (dim: {vector_size})")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection {self.collection_name}: {e}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """
        Upserts document chunks and their dense embeddings into Qdrant.
        """
        if self.client is None or rest_models is None or not chunks:
            return True

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count and embeddings count must match.")

        dim = len(embeddings[0])
        self.ensure_collection(vector_size=dim)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            # Deterministic UUID from chunk_id
            chunk_id = chunk["chunk_id"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            payload = {
                "document_id": str(chunk.get("document_id")),
                "document_name": chunk.get("document_name"),
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("chunk_index", i),
                "page_number": chunk.get("page_number", 1),
                "all_pages": chunk.get("all_pages", [chunk.get("page_number", 1)]),
                "chunk_text": chunk.get("chunk_text", ""),
                "section_title": chunk.get("section_title", "General"),
                "uploaded_at": chunk.get("uploaded_at", ""),
            }

            points.append(rest_models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        try:
            # Batch upsert
            batch_size = 100
            for start in range(0, len(points), batch_size):
                batch = points[start:start + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            logger.info(f"Successfully upserted {len(points)} chunks into Qdrant collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert chunks into Qdrant: {e}")
            raise

    def search_dense(
        self,
        query_vector: List[float],
        limit: int = 20,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes dense vector similarity search in Qdrant with optional document_id filter.
        """
        if self.client is None or rest_models is None:
            return []

        search_filter = None
        if document_id:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=str(document_id))
                    )
                ]
            )

        try:
            # Check if search method or query_points is present
            if hasattr(self.client, "search"):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
            elif hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True
                )
                results = res.points
            else:
                results = []

            formatted = []
            for hit in results:
                payload = getattr(hit, 'payload', {}) or {}
                score = getattr(hit, 'score', 0.0)
                formatted.append({
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "document_name": payload.get("document_name"),
                    "page_number": payload.get("page_number", 1),
                    "all_pages": payload.get("all_pages", [payload.get("page_number", 1)]),
                    "chunk_text": payload.get("chunk_text", ""),
                    "section_title": payload.get("section_title", ""),
                    "score": float(score),
                    "retrieval_source": "dense"
                })
            return formatted
        except Exception as e:
            if "not found" in str(e).lower():
                logger.info(f"Qdrant collection {self.collection_name} does not exist yet. Returning empty results.")
            else:
                logger.error(f"Error during Qdrant dense search: {e}")
            return []

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes all vector points associated with a specific document_id.
        """
        if self.client is None or rest_models is None:
            return True

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=str(document_id))
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted vector points for document_id={document_id} from Qdrant.")
            return True
        except Exception as e:
            if "not found" in str(e).lower():
                return True
            logger.error(f"Failed to delete document {document_id} from Qdrant: {e}")
            return False
