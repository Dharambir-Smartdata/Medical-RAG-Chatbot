"""
Document ingestion pipeline service.
Coordinates PDF extraction, chunking, database persistence,
dense embedding calculation, and Qdrant vector indexing.
"""

import os
import logging
from django.utils import timezone
from django.db import transaction
from apps.documents.models import Document, DocumentChunk
from services.pdf_processor import PDFProcessor, PDFProcessingError
from services.chunking_service import ChunkingService
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


def process_document_pipeline(document_id: str) -> bool:
    """
    Executes the ingestion pipeline for a given Document instance:
    1. Mark status as 'Processing'
    2. Extract text & metadata using PyMuPDF
    3. Generate chunks preserving medical terminology & paragraph boundaries
    4. Save DocumentChunk records to database
    5. Compute batch embeddings via SentenceTransformers
    6. Upsert vector points to Qdrant
    7. Mark status as 'Completed'
    """
    try:
        doc = Document.objects.get(id=document_id)
    except Document.DoesNotExist:
        logger.error(f"Document {document_id} not found.")
        return False

    doc.processing_status = 'Processing'
    doc.error_message = ''
    doc.save(update_fields=['processing_status', 'error_message'])

    try:
        file_path = doc.file_path.path if hasattr(doc.file_path, 'path') else str(doc.file_path)
        if not os.path.exists(file_path):
            raise PDFProcessingError(f"Uploaded file not found at path: {file_path}")

        # 1. Extraction
        extraction = PDFProcessor.extract_text_and_metadata(file_path)
        pages_data = extraction["pages"]
        total_pages = extraction["total_pages"]

        # 2. Chunking
        chunker = ChunkingService()
        chunks_data = chunker.chunk_document(
            document_id=str(doc.id),
            document_name=doc.document_name,
            pages_data=pages_data,
            upload_timestamp=doc.uploaded_at.isoformat()
        )

        if not chunks_data:
            raise PDFProcessingError("No extractable chunks produced from document.")

        # 3. Save Chunks to Database
        with transaction.atomic():
            # Clear any existing chunks for this document
            doc.chunks.all().delete()

            chunk_records = []
            chunk_texts = []
            for item in chunks_data:
                chunk_records.append(DocumentChunk(
                    chunk_id=item["chunk_id"],
                    document=doc,
                    chunk_index=item["chunk_index"],
                    page_number=item["page_number"],
                    all_pages=item.get("all_pages", [item["page_number"]]),
                    chunk_text=item["chunk_text"],
                    section_title=item.get("section_title", "General"),
                    metadata_json={
                        "token_estimate": item.get("token_estimate", 0),
                        "document_name": doc.document_name
                    }
                ))
                chunk_texts.append(item["chunk_text"])

            DocumentChunk.objects.bulk_create(chunk_records)

        # 4. Dense Embedding Generation
        embedder = EmbeddingService()
        embeddings = embedder.embed_texts(chunk_texts)

        # 5. Upsert to Qdrant
        qdrant = QdrantService()
        qdrant.upsert_chunks(chunks=chunks_data, embeddings=embeddings)

        # 6. Complete Document Record
        doc.total_pages = total_pages
        doc.total_chunks = len(chunks_data)
        doc.processing_status = 'Completed'
        doc.processed_at = timezone.now()
        doc.error_message = ''
        doc.save()

        logger.info(f"Successfully processed document {doc.document_name} ({len(chunks_data)} chunks).")
        return True

    except Exception as e:
        logger.error(f"Failed processing document {document_id}: {e}", exc_info=True)
        doc.processing_status = 'Failed'
        doc.error_message = str(e)
        doc.save(update_fields=['processing_status', 'error_message'])
        return False
