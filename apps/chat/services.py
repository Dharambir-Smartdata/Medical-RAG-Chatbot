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
    is_out_of_scope_query,
    is_conversational_follow_up
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

        # 3. Check Conversation History & Follow-Up Context
        past_messages = list(session.messages.order_by('created_at'))
        is_follow_up = bool(past_messages) and is_conversational_follow_up(question)

        last_user_msg = next((m for m in reversed(past_messages) if m.role == 'user'), None)
        last_assistant_msg = next((m for m in reversed(past_messages) if m.role == 'assistant'), None)

        conversation_history_str = ""
        if past_messages:
            history_lines = []
            for msg in past_messages[-6:]:
                role_label = "User" if msg.role == 'user' else "Assistant"
                history_lines.append(f"{role_label}: {msg.message.strip()}")
            conversation_history_str = "\n".join(history_lines)

        search_query = f"{last_user_msg.message} {question}" if (is_follow_up and last_user_msg) else question

        # Retrieve corpus chunks for BM25 sparse search strictly from current user's documents
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
            query=search_query,
            corpus_chunks=corpus_chunks,
            document_id=target_doc_id,
            fusion_limit=15
        )

        # If follow-up, ensure any chunks previously cited in last assistant message are present
        if is_follow_up and last_assistant_msg and last_assistant_msg.citations:
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

        # 5. Question Relevance & Context Sufficiency Check (Pre-Rerank)
        if not GuardrailsService.check_context_sufficiency(fused_candidates, user_question=question, is_follow_up=is_follow_up):
            if is_out_of_scope_query(question):
                insufficient_msg = OUT_OF_CONTEXT_MESSAGE
                refusal_status = "out_of_scope"
            else:
                insufficient_msg = INSUFFICIENT_INFO_MESSAGE
                refusal_status = "insufficient_context"

            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(session=session, role='assistant', message=insufficient_msg, citations=[])
            return {
                "answer": insufficient_msg,
                "citations": [],
                "status": refusal_status
            }

        # 6. Cross-Encoder Reranking & Score Threshold Validation
        top_chunks = self.reranker.rerank(
            query=search_query if is_follow_up else question,
            candidates=fused_candidates,
            top_k=6
        )

        threshold = getattr(settings, 'RERANK_SCORE_THRESHOLD', -8.5)
        top_score = top_chunks[0].get('rerank_score', -99) if top_chunks else -99

        if top_score < threshold and not is_follow_up:
            logger.info(f"Top rerank score {top_score:.4f} is below threshold {threshold} for query '{question}'")
            if is_out_of_scope_query(question):
                refusal_msg = OUT_OF_CONTEXT_MESSAGE
                refusal_status = "out_of_scope"
            else:
                refusal_msg = INSUFFICIENT_INFO_MESSAGE
                refusal_status = "insufficient_context"

            ChatMessage.objects.create(session=session, role='user', message=question)
            ChatMessage.objects.create(session=session, role='assistant', message=refusal_msg, citations=[])
            return {
                "answer": refusal_msg,
                "citations": [],
                "status": refusal_status
            }

        # Filter out chunks below relevance threshold
        filtered_chunks = [c for c in top_chunks if c.get('rerank_score', -99) >= threshold]
        if not filtered_chunks:
            filtered_chunks = top_chunks[:2]

        # 7. Specific Evidence Validation
        q_lower = question.lower()
        if any(term in q_lower for term in ['blood pressure', 'bp ']):
            combined_evidence = " ".join(c.get('chunk_text', '') for c in filtered_chunks).lower()
            if not any(bp_term in combined_evidence for bp_term in ['blood pressure', 'bp', 'mmhg']):
                insufficient_msg = INSUFFICIENT_INFO_MESSAGE
                ChatMessage.objects.create(session=session, role='user', message=question)
                ChatMessage.objects.create(session=session, role='assistant', message=insufficient_msg, citations=[])
                return {
                    "answer": insufficient_msg,
                    "citations": [],
                    "status": "insufficient_context"
                }

        # 8. Prepare Citation Identifiers and Structured Context
        context_str, metadata_str, source_map = CitationService.prepare_citations_context(filtered_chunks)

        # Merge citations from previous assistant turn if follow-up
        if is_follow_up and last_assistant_msg and last_assistant_msg.citations:
            for cit in last_assistant_msg.citations:
                sid = cit.get("source_id")
                if sid and sid not in source_map:
                    source_map[sid] = cit

        # 9. Call Gemini API
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

        # 10. Output Guardrails and Citation Validation
        guarded_answer, is_sufficient = GuardrailsService.enforce_output_guardrails(
            raw_output=raw_answer,
            retrieved_chunks=filtered_chunks,
            user_question=question
        )

        final_answer, verified_citations = CitationService.extract_and_validate_citations(
            answer_text=guarded_answer,
            source_map=source_map
        )

        # If follow-up reformatting produced an answer without explicit brackets, retain prior citations
        if is_follow_up and not verified_citations and last_assistant_msg and last_assistant_msg.citations:
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
            "status": "success",
            "retrieved_chunk_count": len(top_chunks)
        }
