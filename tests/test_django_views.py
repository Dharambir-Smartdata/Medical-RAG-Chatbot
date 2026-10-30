import json
import io
from unittest.mock import patch
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from apps.documents.models import Document, DocumentChunk
from apps.chat.models import ChatSession, ChatMessage

User = get_user_model()


class TestDjangoViews(TestCase):
    """
    Automated tests for Django views, APIs, and document isolation.
    """

    def setUp(self):
        self.client = Client()
        self.user1 = User.objects.create_user(username='dr_alice', password='password123')
        self.user2 = User.objects.create_user(username='dr_bob', password='password123')

        # Create a sample document for user1
        self.doc1 = Document.objects.create(
            user=self.user1,
            document_name="patient_report_alice.pdf",
            file_hash="hash_alice_123",
            total_pages=3,
            total_chunks=6,
            processing_status="Completed"
        )
        self.chunk1 = DocumentChunk.objects.create(
            chunk_id="chunk_alice_001",
            document=self.doc1,
            chunk_index=0,
            page_number=1,
            chunk_text="Patient has diagnosed mild hypertension. Recommended daily exercise.",
            section_title="Diagnosis"
        )

        # Create a sample document for user2
        self.doc2 = Document.objects.create(
            user=self.user2,
            document_name="patient_report_bob.pdf",
            file_hash="hash_bob_456",
            total_pages=2,
            total_chunks=4,
            processing_status="Completed"
        )

    def test_document_listing_isolation(self):
        # Authenticated as user1
        self.client.force_login(self.user1)
        response = self.client.get('/documents/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        doc_names = [d["document_name"] for d in data["documents"]]
        self.assertIn("patient_report_alice.pdf", doc_names)
        self.assertNotIn("patient_report_bob.pdf", doc_names)

        # Authenticated as user2
        self.client.force_login(self.user2)
        response2 = self.client.get('/documents/')
        data2 = response2.json()
        doc_names2 = [d["document_name"] for d in data2["documents"]]
        self.assertIn("patient_report_bob.pdf", doc_names2)
        self.assertNotIn("patient_report_alice.pdf", doc_names2)

    def test_document_detail_view(self):
        self.client.force_login(self.user1)
        response = self.client.get(f'/documents/{self.doc1.id}/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], str(self.doc1.id))
        self.assertEqual(len(data["chunks"]), 1)
        self.assertEqual(data["chunks"][0]["chunk_id"], "chunk_alice_001")

    def test_document_deletion_view(self):
        self.client.force_login(self.user1)
        response = self.client.delete(f'/documents/{self.doc1.id}/delete/')
        self.assertEqual(response.status_code, 200)
        self.assertFalse(Document.objects.filter(id=self.doc1.id).exists())
        self.assertFalse(DocumentChunk.objects.filter(chunk_id="chunk_alice_001").exists())

    def test_create_new_chat_session(self):
        self.client.force_login(self.user1)
        response = self.client.post('/chat/new/')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("session_id", data)
        self.assertTrue(ChatSession.objects.filter(id=data["session_id"]).exists())

    @patch('services.gemini_service.GeminiService.generate_answer')
    def test_submit_chat_question(self, mock_gemini):
        mock_gemini.return_value = "The document mentions mild hypertension [S1]."

        self.client.force_login(self.user1)
        payload = {
            "question": "What is the diagnosis?",
            "document_id": str(self.doc1.id)
        }
        response = self.client.post(
            '/chat/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("hypertension", data["answer"].lower())
        self.assertEqual(data["status"], "success")

    def test_chat_no_documents_response(self):
        # Create a new user with no uploaded documents
        new_user = User.objects.create_user(username='dr_empty', password='password123')
        self.client.force_login(new_user)

        payload = {"question": "Any findings?"}
        response = self.client.post(
            '/chat/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Please upload a PDF document", data["answer"])
        self.assertEqual(data["status"], "no_documents")

    @patch('services.gemini_service.GeminiService.generate_answer')
    def test_edit_chat_question(self, mock_gemini):
        mock_gemini.return_value = "The patient has hypertension [S1]."

        self.client.force_login(self.user1)
        # 1. Send initial question
        resp1 = self.client.post(
            '/chat/',
            data=json.dumps({
                "question": "What is the diagnosis?",
                "document_id": str(self.doc1.id)
            }),
            content_type='application/json'
        )
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        session_id = data1["session_id"]
        old_user_msg_id = data1["user_message_id"]

        # 2. Send edited question replacing the old one
        mock_gemini.return_value = "The document recommends daily exercise [S1]."
        resp2 = self.client.post(
            '/chat/',
            data=json.dumps({
                "question": "What exercise was recommended?",
                "session_id": session_id,
                "document_id": str(self.doc1.id),
                "edit_message_id": old_user_msg_id
            }),
            content_type='application/json'
        )
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertIn("daily exercise", data2["answer"])
        self.assertNotEqual(data2["user_message_id"], old_user_msg_id)
        # Verify old message was deleted from session
        session = ChatSession.objects.get(id=session_id)
        self.assertFalse(session.messages.filter(id=old_user_msg_id).exists())

