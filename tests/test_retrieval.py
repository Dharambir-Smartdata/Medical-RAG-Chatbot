from django.test import TestCase
from services.sparse_retrieval_service import SparseRetrievalService
from services.hybrid_retrieval_service import HybridRetrievalService


class TestRetrieval(TestCase):
    """
    Automated tests for Sparse BM25, Dense retrieval, and Reciprocal Rank Fusion.
    """

    def setUp(self):
        self.sparse_service = SparseRetrievalService()
        self.sample_chunks = [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_1",
                "document_name": "cardiology_report.pdf",
                "page_number": 2,
                "chunk_text": "Patient has acute myocardial infarction. Prescribed Aspirin 81 mg and Atorvastatin 40 mg."
            },
            {
                "chunk_id": "chunk_002",
                "document_id": "doc_1",
                "document_name": "cardiology_report.pdf",
                "page_number": 3,
                "chunk_text": "Follow-up echocardiogram shows ejection fraction of 50%. Patient reports no chest pain."
            },
            {
                "chunk_id": "chunk_003",
                "document_id": "doc_2",
                "document_name": "endocrinology_note.pdf",
                "page_number": 1,
                "chunk_text": "Patient with Type 2 Diabetes Mellitus with ICD-10 code E11.9. Prescribed Metformin 500 mg."
            }
        ]

    def test_sparse_bm25_retrieval(self):
        results = self.sparse_service.search_sparse(
            query="What is the dose of Atorvastatin?",
            corpus_chunks=self.sample_chunks
        )
        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match["chunk_id"], "chunk_001")
        self.assertIn("Atorvastatin 40 mg", top_match["chunk_text"])

    def test_sparse_bm25_metadata_filtering(self):
        # Query matching doc_1 but filtered to doc_2
        results = self.sparse_service.search_sparse(
            query="Atorvastatin",
            corpus_chunks=self.sample_chunks,
            document_id="doc_2"
        )
        self.assertEqual(len(results), 0)

        # Query matching doc_2
        results_doc2 = self.sparse_service.search_sparse(
            query="Metformin",
            corpus_chunks=self.sample_chunks,
            document_id="doc_2"
        )
        self.assertEqual(len(results_doc2), 1)
        self.assertEqual(results_doc2[0]["chunk_id"], "chunk_003")

    def test_reciprocal_rank_fusion(self):
        # List 1: Dense ranked
        dense_ranked = [
            {"chunk_id": "chunk_001", "chunk_text": "Text 1", "retrieval_source": "dense"},
            {"chunk_id": "chunk_002", "chunk_text": "Text 2", "retrieval_source": "dense"},
        ]
        # List 2: Sparse ranked
        sparse_ranked = [
            {"chunk_id": "chunk_002", "chunk_text": "Text 2", "retrieval_source": "sparse"},
            {"chunk_id": "chunk_003", "chunk_text": "Text 3", "retrieval_source": "sparse"},
        ]

        fused = HybridRetrievalService.reciprocal_rank_fusion(
            ranked_lists=[dense_ranked, sparse_ranked],
            rrf_k=60
        )

        # Verify deduplication
        fused_ids = [c["chunk_id"] for c in fused]
        self.assertEqual(len(fused_ids), len(set(fused_ids)))
        self.assertIn("chunk_001", fused_ids)
        self.assertIn("chunk_002", fused_ids)
        self.assertIn("chunk_003", fused_ids)

        # chunk_002 was in both lists, so its combined RRF score should be highest!
        self.assertEqual(fused[0]["chunk_id"], "chunk_002")
        self.assertIn("dense", fused[0]["retrieval_sources"])
        self.assertIn("sparse", fused[0]["retrieval_sources"])
