from django.test import TestCase
from services.chunking_service import ChunkingService


class TestChunkingService(TestCase):
    """
    Automated tests for configurable chunking pipeline.
    """

    def setUp(self):
        self.chunker = ChunkingService(chunk_size=100, chunk_overlap=30)

    def test_chunk_creation_and_metadata(self):
        pages_data = [
            {
                "page_number": 1,
                "text": "Diagnosis: Hypertension and Type 2 Diabetes.\n\nPatient is prescribed Lisinopril 20 mg daily and Metformin 500 mg b.i.d."
            },
            {
                "page_number": 2,
                "text": "Laboratory Findings: HbA1c is 7.4%. Serum creatinine is 0.9 mg/dL.\n\nPlan: Continue current regimen and recheck in 3 months."
            }
        ]

        chunks = self.chunker.chunk_document(
            document_id="doc_test_101",
            document_name="clinical_note.pdf",
            pages_data=pages_data,
            upload_timestamp="2026-10-28T12:00:00"
        )

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]

        # Verify Metadata
        self.assertEqual(first_chunk["document_id"], "doc_test_101")
        self.assertEqual(first_chunk["document_name"], "clinical_note.pdf")
        self.assertEqual(first_chunk["page_number"], 1)
        self.assertEqual(first_chunk["chunk_index"], 0)
        self.assertIn("chunk_id", first_chunk)
        self.assertTrue(first_chunk["chunk_id"].startswith("doc_test_101_chunk_"))

        # Verify clinical term preservation
        all_text = " ".join(c["chunk_text"] for c in chunks)
        self.assertIn("Lisinopril 20 mg", all_text)
        self.assertIn("Metformin 500 mg", all_text)
        self.assertIn("HbA1c is 7.4%", all_text)

    def test_multi_page_traceability(self):
        pages_data = [
            {"page_number": 1, "text": "Short intro on page one."},
            {"page_number": 2, "text": "Continuous clinical observation on page two."}
        ]
        # Large chunk size so both pages combine into one chunk
        chunker = ChunkingService(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document("doc_multi", "doc.pdf", pages_data)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn(1, chunks[0]["all_pages"])
        self.assertIn(2, chunks[0]["all_pages"])

    def test_empty_pages_handling(self):
        chunks = self.chunker.chunk_document("doc_empty", "empty.pdf", [])
        self.assertEqual(chunks, [])
