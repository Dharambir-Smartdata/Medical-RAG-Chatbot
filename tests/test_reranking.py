from django.test import TestCase
from services.reranking_service import RerankingService


class TestReranking(TestCase):
    """
    Automated tests for Cross-Encoder Reranking service.
    """

    def setUp(self):
        self.reranker = RerankingService()
        self.candidates = [
            {
                "chunk_id": "chunk_A",
                "document_name": "lab_results.pdf",
                "page_number": 4,
                "chunk_text": "Complete Blood Count: WBC is 6.5, Hemoglobin is 14.2 g/dL, Platelets 250k.",
                "rrf_score": 0.02
            },
            {
                "chunk_id": "chunk_B",
                "document_name": "medications.pdf",
                "page_number": 1,
                "chunk_text": "Prescription details: Lisinopril 10 mg oral tablet once daily for blood pressure.",
                "rrf_score": 0.05
            },
            {
                "chunk_id": "chunk_C",
                "document_name": "billing.pdf",
                "page_number": 1,
                "chunk_text": "Hospital administrative billing statement and insurance copay receipt.",
                "rrf_score": 0.01
            }
        ]

    def test_reranker_selection_and_limit(self):
        top_chunks = self.reranker.rerank(
            query="What is the blood pressure medication?",
            candidates=self.candidates,
            top_k=2
        )

        self.assertEqual(len(top_chunks), 2)
        # Verify metadata preservation
        for c in top_chunks:
            self.assertIn("document_name", c)
            self.assertIn("page_number", c)
            self.assertIn("chunk_id", c)

    def test_empty_candidates_handling(self):
        result = self.reranker.rerank(query="Some query", candidates=[], top_k=5)
        self.assertEqual(result, [])
