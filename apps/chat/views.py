"""
Views and API endpoints for Chat Management and Medical RAG execution.
"""

import json
import logging
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from apps.chat.models import ChatSession, ChatMessage
from apps.chat.services import ChatRAGOrchestrator
from apps.documents.models import Document

logger = logging.getLogger(__name__)


def _get_user_sessions(request):
    """Retrieves chat sessions for current user or session."""
    if request.user.is_authenticated:
        return ChatSession.objects.filter(user=request.user)
    return ChatSession.objects.filter(user__isnull=True)


@ensure_csrf_cookie
def chat_home(request):
    """Renders the primary Medical AI Assistant Chat UI."""
    return render(request, 'chat.html')


_orchestrator = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatRAGOrchestrator()
    return _orchestrator


@require_http_methods(["POST"])
def submit_question(request):
    """
    POST /chat/
    Processes user query through full Medical RAG pipeline.
    Payload:
    {
        "question": "...",
        "session_id": "uuid-optional",
        "document_id": "all" or "doc-uuid"
    }
    """
    try:
        data = json.loads(request.body.decode('utf-8'))
    except Exception:
        data = request.POST

    question = (data.get('question') or '').strip()
    session_id = data.get('session_id')
    document_id = data.get('document_id')

    if not question:
        return JsonResponse({"error": "Question cannot be empty."}, status=400)

    # Resolve or create ChatSession
    session = None
    if session_id:
        session = _get_user_sessions(request).filter(id=session_id).first()

    if not session:
        user = request.user if request.user.is_authenticated else None
        session = ChatSession.objects.create(
            user=user,
            session_name="New Medical Consultation"
        )

    edit_message_id = data.get('edit_message_id')
    if edit_message_id and session:
        target_msg = session.messages.filter(id=edit_message_id).first()
        if target_msg:
            # Delete the edited message and any subsequent messages in this session
            session.messages.filter(created_at__gte=target_msg.created_at).delete()

    # Run Cached RAG Orchestrator
    orchestrator = get_orchestrator()
    user = request.user if request.user.is_authenticated else None
    result = orchestrator.answer_question(
        question=question,
        session=session,
        user=user,
        document_id=document_id
    )

    user_msg = session.messages.filter(role='user').last()
    asst_msg = session.messages.filter(role='assistant').last()

    return JsonResponse({
        "session_id": str(session.id),
        "session_name": session.session_name,
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "status": result.get("status", "success"),
        "error_type": result.get("error_type", None),
        "retryable": result.get("retryable", False),
        "user_message_id": user_msg.id if user_msg else None,
        "assistant_message_id": asst_msg.id if asst_msg else None
    })


@require_http_methods(["GET"])
def get_chat_history(request):
    """
    GET /chat/history/
    Lists previous chat sessions for current user.
    """
    sessions = _get_user_sessions(request)
    data = []
    for s in sessions:
        last_msg = s.messages.last()
        data.append({
            "id": str(s.id),
            "session_name": s.session_name,
            "created_at": s.created_at.strftime("%b %d, %H:%M"),
            "updated_at": s.updated_at.strftime("%b %d, %H:%M"),
            "message_count": s.messages.count(),
            "last_message": last_msg.message[:60] if last_msg else ""
        })
    return JsonResponse({"sessions": data})


@require_http_methods(["POST"])
def create_new_session(request):
    """
    POST /chat/new/
    Creates a new empty chat session.
    """
    user = request.user if request.user.is_authenticated else None
    session = ChatSession.objects.create(
        user=user,
        session_name="New Medical Consultation"
    )
    return JsonResponse({
        "session_id": str(session.id),
        "session_name": session.session_name
    })


@require_http_methods(["GET"])
def get_session_messages(request, session_id):
    """
    GET /chat/session/<session_id>/
    Retrieves full transcript and citations for a specific chat session.
    """
    session = get_object_or_404(_get_user_sessions(request), id=session_id)
    messages_data = []
    for msg in session.messages.all():
        messages_data.append({
            "id": msg.id,
            "role": msg.role,
            "message": msg.message,
            "citations": msg.citations,
            "created_at": msg.created_at.strftime("%H:%M")
        })

    return JsonResponse({
        "session_id": str(session.id),
        "session_name": session.session_name,
        "messages": messages_data
    })


@require_http_methods(["DELETE", "POST"])
def delete_session(request, session_id):
    """
    DELETE /chat/session/<session_id>/
    Deletes a chat session and associated messages.
    """
    session = get_object_or_404(_get_user_sessions(request), id=session_id)
    session.delete()
    return JsonResponse({"success": True, "message": "Chat session deleted successfully."})


@require_http_methods(["GET"])
def check_model_status(request):
    """
    GET /chat/model-status/
    Checks whether transformer embedding and reranker models are downloaded/initialized.
    """
    from services.embedding_service import EmbeddingService
    from services.reranking_service import RerankingService

    try:
        emb = EmbeddingService()
        rerank = RerankingService()
        is_ready = (emb._model is not None) and (rerank._model is not None)
        return JsonResponse({
            "ready": is_ready,
            "embedding_loaded": emb._model is not None,
            "reranker_loaded": rerank._model is not None,
            "embedding_model": getattr(emb, 'model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            "reranker_model": getattr(rerank, 'model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        })
    except Exception as e:
        logger.warning(f"Model status check encountered error: {e}")
        return JsonResponse({
            "ready": False,
            "error": str(e)
        })

