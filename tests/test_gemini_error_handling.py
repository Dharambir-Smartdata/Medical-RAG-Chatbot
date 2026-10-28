"""
Comprehensive unit and integration tests for Gemini API error handling,
transient retry logic, and strict RAG error responses.
Covers all 15 test scenarios requested in the specification:
1. Gemini successfully answers a relevant PDF question.
2. Gemini returns 503 UNAVAILABLE.
3. Gemini returns 429 RESOURCE_EXHAUSTED.
4. Gemini returns 401 UNAUTHENTICATED.
5. Gemini returns 403 PERMISSION_DENIED.
6. Gemini returns 400 INVALID_ARGUMENT.
7. Gemini returns 500 INTERNAL.
8. Gemini request times out.
9. Network connection fails.
10. Gemini returns an empty response.
11. Relevant PDF information is missing.
12. Retrieved chunks are unrelated.
13. Different questions produce question-specific answers.
14. Retry succeeds after temporary model failure.
15. Retry fails and the appropriate error is displayed.
"""

from unittest.mock import patch, MagicMock
from django.test import TestCase
from django.contrib.auth import get_user_model
import httpx
from google.genai import errors

from apps.documents.models import Document, DocumentChunk
from apps.chat.models import ChatSession
from apps.chat.services import ChatRAGOrchestrator
from services.gemini_service import GeminiService, GeminiServiceException

User = get_user_model()


class TestGeminiErrorHandlingAndRetries(TestCase):
    """
    Validates intelligent LLM error handling, retries, and honest non-dummy error responses.
    """

    def setUp(self):
        self.orchestrator = ChatRAGOrchestrator()
        self.gemini_service = self.orchestrator.gemini_service
        self.user = User.objects.create_user(username='test_patient', password='password123')

        # Create a document for the user
        self.doc = Document.objects.create(
            user=self.user,
            document_name="discharge_summary.pdf",
            file_hash="hash_123",
            total_pages=1,
            total_chunks=1,
            processing_status="Completed"
        )
        self.chunk = DocumentChunk.objects.create(
            chunk_id="chunk_01",
            document=self.doc,
            chunk_index=0,
            page_number=1,
            chunk_text=(
                "Encounter Summary: Outpatient evaluation on 2026-10-20. "
                "Chief concern and reported symptoms: intermittent fatigue and increased thirst. "
                "Vital signs: Blood pressure 128/82 mmHg; pulse 78 beats/min. "
                "HbA1c: 6.4 % (Reference: 4.0–5.6 %, High)."
            ),
            section_title="Encounter Summary"
        )
        self.session = ChatSession.objects.create(user=self.user, session_name="Test Consultation")

        # Mock time.sleep to make retry tests instantaneous
        self.sleep_patch = patch('time.sleep')
        self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def test_01_gemini_successfully_answers(self):
        """1. Gemini successfully answers a relevant PDF question."""
        with patch.object(self.gemini_service.client.models, 'generate_content') as mock_call:
            mock_resp = MagicMock()
            mock_resp.text = "The patient reported intermittent fatigue and increased thirst [S1]."
            mock_call.return_value = mock_resp

            resp = self.orchestrator.answer_question(
                question="What symptoms did the patient report?",
                session=self.session,
                user=self.user,
                document_id=str(self.doc.id)
            )

            self.assertEqual(resp["status"], "success")
            self.assertIn("fatigue", resp["answer"].lower())
            self.assertGreater(len(resp["citations"]), 0)

    def test_02_gemini_503_unavailable(self):
        """2. Gemini returns 503 UNAVAILABLE: Returns accurate message without dummy medical fallback."""
        err = errors.ServerError(503, {'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': 'high demand'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's HbA1c value?",
                session=self.session,
                user=self.user,
                document_id=str(self.doc.id)
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "service_unavailable")
            self.assertTrue(resp.get("retryable"))
            self.assertIn("temporarily unavailable due to high demand", resp["answer"])
            # Verify NO fake medical data returned
            self.assertNotIn("Fasting plasma glucose", resp["answer"])
            self.assertNotIn("Creatinine", resp["answer"])

    def test_03_gemini_429_quota_exhausted(self):
        """3. Gemini returns 429 RESOURCE_EXHAUSTED."""
        err = errors.ClientError(429, {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'message': 'daily limit'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's HbA1c value?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "quota_exceeded")
            self.assertFalse(resp.get("retryable"))
            self.assertIn("usage limit has been reached", resp["answer"])

    def test_04_gemini_401_unauthenticated(self):
        """4. Gemini returns 401 UNAUTHENTICATED."""
        err = errors.ClientError(401, {'error': {'code': 401, 'status': 'UNAUTHENTICATED', 'message': 'bad key'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's HbA1c value?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "authentication_error")
            self.assertFalse(resp.get("retryable"))
            self.assertIn("authentication failed", resp["answer"])

    def test_05_gemini_403_permission_denied(self):
        """5. Gemini returns 403 PERMISSION_DENIED."""
        err = errors.ClientError(403, {'error': {'code': 403, 'status': 'PERMISSION_DENIED', 'message': 'blocked'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "authentication_error")
            self.assertIn("authentication failed", resp["answer"])

    def test_06_gemini_400_invalid_argument(self):
        """6. Gemini returns 400 INVALID_ARGUMENT."""
        err = errors.ClientError(400, {'error': {'code': 400, 'status': 'INVALID_ARGUMENT', 'message': 'bad params'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "invalid_request")
            self.assertIn("invalid or unsupported", resp["answer"])

    def test_07_gemini_500_internal_error(self):
        """7. Gemini returns 500 INTERNAL."""
        err = errors.ServerError(500, {'error': {'code': 500, 'status': 'INTERNAL', 'message': 'server crash'}})
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "internal_error")
            self.assertIn("encountered an internal error", resp["answer"])

    def test_08_gemini_timeout(self):
        """8. Gemini request times out."""
        err = httpx.ReadTimeout("The read operation timed out")
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "timeout")
            self.assertIn("took too long to respond", resp["answer"])

    def test_09_gemini_network_connection_fails(self):
        """9. Network connection fails."""
        err = OSError("[Errno 101] Network is unreachable")
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=err):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "network_error")
            self.assertIn("Unable to connect to the AI service", resp["answer"])

    def test_10_gemini_empty_response(self):
        """10. Gemini returns an empty or whitespace response."""
        empty_resp = MagicMock()
        empty_resp.text = "   "
        with patch.object(self.gemini_service.client.models, 'generate_content', return_value=empty_resp):
            resp = self.orchestrator.answer_question(
                question="What was the patient's blood pressure?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp["status"], "llm_error")
            self.assertEqual(resp.get("error_type"), "empty_response")
            self.assertIn("returned an empty response", resp["answer"])

    def test_11_relevant_pdf_information_is_missing(self):
        """11. Relevant PDF information is missing from uploaded documents."""
        resp = self.orchestrator.answer_question(
            question="What was the patient's potassium value?",
            session=self.session,
            user=self.user
        )
        self.assertEqual(resp["status"], "insufficient_context")
        self.assertIn("could not find sufficient information in your uploaded documents", resp["answer"])

    def test_12_retrieved_chunks_unrelated(self):
        """12. Retrieved chunks are unrelated / out of scope."""
        resp = self.orchestrator.answer_question(
            question="Who was the prime minister of the United Kingdom in 1945?",
            session=self.session,
            user=self.user
        )
        self.assertIn(resp["status"], ["out_of_scope", "rejected"])
        self.assertIn("can only answer questions related to your uploaded pdf documents", resp["answer"].lower())

    def test_13_different_questions_produce_question_specific_answers(self):
        """13. Different questions produce question-specific answers without reusing old ones."""
        with patch.object(self.gemini_service.client.models, 'generate_content') as mock_call:
            mock_resp_1 = MagicMock()
            mock_resp_1.text = "The blood pressure was 128/82 mmHg [S1]."
            mock_resp_2 = MagicMock()
            mock_resp_2.text = "The pulse was 78 beats/min [S1]."
            mock_call.side_effect = [mock_resp_1, mock_resp_2]

            resp_1 = self.orchestrator.answer_question(
                question="What was the blood pressure?",
                session=self.session,
                user=self.user
            )
            resp_2 = self.orchestrator.answer_question(
                question="What was the pulse?",
                session=self.session,
                user=self.user
            )

            self.assertEqual(resp_1["status"], "success")
            self.assertEqual(resp_2["status"], "success")
            self.assertIn("128/82", resp_1["answer"])
            self.assertIn("78", resp_2["answer"])
            self.assertNotEqual(resp_1["answer"], resp_2["answer"])

    def test_14_retry_succeeds_after_temporary_failure(self):
        """14. Retry succeeds after temporary 503 model failure."""
        err_503 = errors.ServerError(503, {'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': 'high demand'}})
        success_resp = MagicMock()
        success_resp.text = "HbA1c value was 6.4% [S1]."

        # Fail once with 503, succeed on attempt 2
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=[err_503, success_resp]):
            with patch('time.sleep'):  # Avoid actual test delay
                resp = self.orchestrator.answer_question(
                    question="What was the patient's HbA1c?",
                    session=self.session,
                    user=self.user
                )

                self.assertEqual(resp["status"], "success")
                self.assertIn("6.4%", resp["answer"])

    def test_15_retry_fails_and_appropriate_error_is_displayed(self):
        """15. Retry fails across all attempts and appropriate human-readable error is displayed."""
        err_503 = errors.ServerError(503, {'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': 'high demand'}})

        # Fail all attempts (attempt 0, attempt 1, attempt 2)
        with patch.object(self.gemini_service.client.models, 'generate_content', side_effect=[err_503, err_503, err_503]):
            with patch('time.sleep'):
                resp = self.orchestrator.answer_question(
                    question="What was the patient's HbA1c?",
                    session=self.session,
                    user=self.user
                )

                self.assertEqual(resp["status"], "llm_error")
                self.assertEqual(resp.get("error_type"), "service_unavailable")
                self.assertIn("temporarily unavailable due to high demand", resp["answer"])
                # Zero fake citations or answers
                self.assertEqual(len(resp.get("citations", [])), 0)
