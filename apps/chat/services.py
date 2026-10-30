"""
Chat RAG Service.
Coordinates the end-to-end Medical RAG workflow:
Input Guardrails -> Hybrid Retrieval -> RRF Fusion -> Reranking -> Gemini LLM -> Output Guardrails & Citations.
"""

import logging
from typing import Dict, Any, Optional
from django.conf import settings
from django.contrib.auth import get_user_model
from apps.documents.models import Document, DocumentChunk
from apps.chat.models import ChatSession, ChatMessage
from services.guardrails_service import (
    GuardrailsService,
    STANDARD_DISCLAIMER,
    OUT_OF_CONTEXT_MESSAGE,
    INSUFFICIENT_INFO_MESSAGE,
    NO_DOCUMENTS_MESSAGE,
    is_out_of_scope_query
)
from services.hybrid_retrieval_service import HybridRetrievalService
from services.reranking_service import RerankingService
from services.citation_service import CitationService
from services.gemini_service import GeminiService, GeminiServiceException

logger = logging.getLogger(__name__)
User = get_user_model()


class ChatRAGOrchestrator:
    """
    End-to-end Medical RAG query orchestrator.
    """

    def __init__(self):
        self.hybrid_retriever = HybridRetrievalService()
        self.reranker = RerankingService()
        self.gemini_service = GeminiService()

    def answer_question(
        self,
        question: str,
        session: ChatSession,
        user=None,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full Medical RAG pipeline and persists the conversation.
        """
        # 1. Input Guardrails
        is_valid, guard_message, category = GuardrailsService.validate_input(question)
        if not is_valid:
            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(session=session, role='assistant', message=guard_message, citations=[])
            return {
                "answer": guard_message,
                "citations": [],
                "status": "rejected",
                "category": category
            }

        # 2. Document Availability & User Isolation Check
        docs_qs = Document.objects.filter(processing_status='Completed')
        if user and user.is_authenticated:
            docs_qs = docs_qs.filter(user=user)
        else:
            docs_qs = docs_qs.filter(user__isnull=True)

        if not docs_qs.exists():
            no_docs_msg = NO_DOCUMENTS_MESSAGE
            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(session=session, role='assistant', message=no_docs_msg, citations=[])
            return {
                "answer": no_docs_msg,
                "citations": [],
                "status": "no_documents"
            }

        # If specific document selected, verify it belongs to user/context
        target_doc_id = None
        if document_id and document_id != 'all':
            target_doc = docs_qs.filter(id=document_id).first()
            if target_doc:
                target_doc_id = str(target_doc.id)

        # 3. Check Conversation History
        past_messages = list(session.messages.order_by('created_at'))
        last_assistant_msg = next((m for m in reversed(past_messages) if m.role == 'assistant'), None)

        conversation_history_str = ""
        if past_messages:
            history_lines = []
            for msg in past_messages[-6:]:
                role_label = "User" if msg.role == 'user' else "Assistant"
                history_lines.append(f"{role_label}: {msg.message.strip()}")
            conversation_history_str = "\n".join(history_lines)

        # Retrieve corpus chunks strictly from current user's documents
        chunks_qs = DocumentChunk.objects.filter(document__in=docs_qs)
        if target_doc_id:
            chunks_qs = chunks_qs.filter(document_id=target_doc_id)

        corpus_chunks = []
        for c in chunks_qs.select_related('document'):
            corpus_chunks.append({
                "chunk_id": c.chunk_id,
                "document_id": str(c.document_id),
                "document_name": c.document.document_name,
                "page_number": c.page_number,
                "all_pages": c.all_pages,
                "chunk_text": c.chunk_text,
                "section_title": c.section_title
            })

        # 4. Hybrid Retrieval (Dense Qdrant + Sparse BM25 + RRF)
        fused_candidates = self.hybrid_retriever.retrieve(
            query=question,
            corpus_chunks=corpus_chunks,
            document_id=target_doc_id,
            fusion_limit=15
        )

        # Retain chunks previously cited in conversation so ongoing discussion has context
        if last_assistant_msg and last_assistant_msg.citations:
            prior_chunk_ids = {c.get("chunk_id") for c in last_assistant_msg.citations if c.get("chunk_id")}
            prior_doc_pages = {(c.get("document_name"), c.get("page_number")) for c in last_assistant_msg.citations}
            existing_ids = {c.get("chunk_id") for c in fused_candidates}
            matching_prior = [
                c for c in corpus_chunks
                if (c.get("chunk_id") in prior_chunk_ids or 
                    (c.get("document_name"), c.get("page_number")) in prior_doc_pages)
                and c.get("chunk_id") not in existing_ids
            ]
            if matching_prior:
                fused_candidates = matching_prior + fused_candidates

        # 5. Cross-Encoder Reranking
        top_chunks = self.reranker.rerank(
            query=question,
            candidates=fused_candidates,
            top_k=6
        )
        if not top_chunks:
            top_chunks = fused_candidates[:6]

        filtered_chunks = top_chunks[:6]

        # 6. Prepare Citation Identifiers and Structured Context
        context_str, metadata_str, source_map = CitationService.prepare_citations_context(filtered_chunks)

        # Merge citations from previous assistant turn if available
        if last_assistant_msg and last_assistant_msg.citations:
            for cit in last_assistant_msg.citations:
                sid = cit.get("source_id")
                if sid and sid not in source_map:
                    source_map[sid] = cit

        # 7. Call Gemini API directly with query, context, and conversation history
        try:
            raw_answer = self.gemini_service.generate_answer(
                user_question=question,
                retrieved_context=context_str,
                source_metadata=metadata_str,
                conversation_history=conversation_history_str
            )
        except GeminiServiceException as e:
            logger.warning(f"Gemini API failure [{e.error_type}]: {e.user_message}")
            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(
                session=session,
                role='assistant',
                message=e.user_message,
                citations=[]
            )
            return {
                "answer": e.user_message,
                "citations": [],
                "status": "llm_error",
                "error_type": e.error_type,
                "retryable": e.retryable
            }
        except Exception as e:
            logger.error(f"Unexpected generation error: {e}", exc_info=True)
            fallback_err = "The AI service encountered an unexpected error. Please try again shortly."
            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(
                session=session,
                role='assistant',
                message=fallback_err,
                citations=[]
            )
            return {
                "answer": fallback_err,
                "citations": [],
                "status": "llm_error",
                "error_type": "unknown_error",
                "retryable": True
            }

        # 8. Output Guardrails and Citation Validation
        guarded_answer, is_sufficient = GuardrailsService.enforce_output_guardrails(
            raw_output=raw_answer,
            retrieved_chunks=filtered_chunks,
            user_question=question
        )

        final_answer, verified_citations = CitationService.extract_and_validate_citations(
            answer_text=guarded_answer,
            source_map=source_map
        )

        # If reformatting produced an answer without explicit brackets, retain prior citations
        if not verified_citations and last_assistant_msg and last_assistant_msg.citations:
            verified_citations = list(last_assistant_msg.citations)

        # Persist Messages to Database
        ChatMessage.objects.create(session=session, role='user', message=question)
        ChatMessage.objects.create(
            session=session,
            role='assistant',
            message=final_answer,
            citations=verified_citations
        )

        # Automatically update session name if it's the first exchange
        if session.messages.filter(role='user').count() <= 1:
            snippet = question.strip()[:40]
            session.session_name = snippet + ("..." if len(question) > 40 else "")
            session.save(update_fields=['session_name', 'updated_at'])

        return {
            "answer": final_answer,
            "citations": verified_citations,
            "status": "success" if is_sufficient else "insufficient_context",
            "retrieved_chunk_count": len(filtered_chunks)
        }
