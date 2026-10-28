"""
Views and API endpoints for Document Management.
Supports multi-PDF upload, listing, details, and deletion with Qdrant synchronization.
"""

import os
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
from django.conf import settings

from apps.documents.models import Document, DocumentChunk
from apps.documents.services import process_document_pipeline
from services.pdf_processor import PDFProcessor
from services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


def _get_user_documents(request):
    """Filters documents for current user or session."""
    if request.user.is_authenticated:
        return Document.objects.filter(user=request.user)
    # Anonymous / dev session handling
    session_key = request.session.session_key
    if not session_key:
        request.session.create()
        session_key = request.session.session_key
    return Document.objects.filter(user__isnull=True)


@ensure_csrf_cookie
def documents_page(request):
    """Renders the HTML documents management page."""
    return render(request, 'documents.html')


@require_http_methods(["GET"])
def list_documents(request):
    """
    GET /documents/
    Returns list of uploaded documents and their processing statuses.
    """
    docs = _get_user_documents(request)
    data = []
    for d in docs:
        data.append({
            "id": str(d.id),
            "document_name": d.document_name,
            "total_pages": d.total_pages,
            "total_chunks": d.total_chunks,
            "processing_status": d.processing_status,
            "error_message": d.error_message,
            "uploaded_at": d.uploaded_at.strftime("%Y-%m-%d %H:%M:%S") if d.uploaded_at else "",
            "processed_at": d.processed_at.strftime("%Y-%m-%d %H:%M:%S") if d.processed_at else None,
        })
    return JsonResponse({"documents": data})


@require_http_methods(["POST"])
def upload_documents(request):
    """
    POST /documents/upload/
    Uploads one or multiple medical PDF documents, validates them,
    and runs the ingestion pipeline.
    """
    files = request.FILES.getlist('files') or request.FILES.getlist('file')
    if not files:
        return JsonResponse({"error": "No files provided in the upload request."}, status=400)

    uploaded_docs = []
    errors = []

    user = request.user if request.user.is_authenticated else None

    for f in files:
        # 1. Validation
        is_valid, err_msg = PDFProcessor.validate_file(f)
        if not is_valid:
            errors.append({"filename": f.name, "error": err_msg})
            continue

        # 2. Hash check for duplicates
        file_hash = PDFProcessor.compute_sha256(f)
        existing = Document.objects.filter(file_hash=file_hash)
        if user:
            existing = existing.filter(user=user)
        else:
            existing = existing.filter(user__isnull=True)

        if existing.exists():
            existing_doc = existing.first()
            # If previous attempt failed, allow re-processing
            if existing_doc.processing_status != 'Failed':
                errors.append({
                    "filename": f.name,
                    "error": f"Document with identical content already exists ({existing_doc.document_name})."
                })
                continue
            else:
                existing_doc.delete()

        # 3. Create Document record
        doc = Document.objects.create(
            user=user,
            document_name=f.name,
            file_path=f,
            file_hash=file_hash,
            processing_status='Uploaded'
        )

        # 4. Trigger ingestion pipeline synchronously
        success = process_document_pipeline(str(doc.id))
        doc.refresh_from_db()

        uploaded_docs.append({
            "id": str(doc.id),
            "document_name": doc.document_name,
            "processing_status": doc.processing_status,
            "total_pages": doc.total_pages,
            "total_chunks": doc.total_chunks,
            "error_message": doc.error_message,
        })

    return JsonResponse({
        "uploaded": uploaded_docs,
        "errors": errors,
        "message": f"Processed {len(uploaded_docs)} document(s)." + (f" Encountered {len(errors)} error(s)." if errors else "")
    }, status=200 if uploaded_docs else 400)


@require_http_methods(["GET"])
def document_detail(request, document_id):
    """
    GET /documents/<document_id>/
    Retrieves full details of a single document including its chunks.
    """
    qs = _get_user_documents(request)
    doc = get_object_or_404(qs, id=document_id)

    chunks = []
    for c in doc.chunks.all():
        chunks.append({
            "chunk_id": c.chunk_id,
            "chunk_index": c.chunk_index,
            "page_number": c.page_number,
            "all_pages": c.all_pages,
            "section_title": c.section_title,
            "chunk_text": c.chunk_text,
            "snippet": c.chunk_text[:150] + "..." if len(c.chunk_text) > 150 else c.chunk_text
        })

    return JsonResponse({
        "id": str(doc.id),
        "document_name": doc.document_name,
        "total_pages": doc.total_pages,
        "total_chunks": doc.total_chunks,
        "processing_status": doc.processing_status,
        "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else "",
        "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
        "chunks": chunks
    })


@require_http_methods(["DELETE", "POST"])
def delete_document(request, document_id):
    """
    DELETE /documents/<document_id>/
    Deletes the document, its chunks from DB, its file from disk,
    and all associated vector points from Qdrant.
    """
    qs = _get_user_documents(request)
    doc = get_object_or_404(qs, id=document_id)

    doc_name = doc.document_name
    doc_id_str = str(doc.id)

    # 1. Delete vectors from Qdrant
    qdrant = QdrantService()
    qdrant.delete_document(doc_id_str)

    # 2. Delete file from storage
    try:
        if doc.file_path and os.path.exists(doc.file_path.path):
            os.remove(doc.file_path.path)
    except Exception as e:
        logger.warning(f"Could not remove physical file for {doc_name}: {e}")

    # 3. Delete DB record (cascades to chunks)
    doc.delete()

    return JsonResponse({
        "success": True,
        "message": f"Document '{doc_name}' and all associated vector records successfully deleted."
    })
