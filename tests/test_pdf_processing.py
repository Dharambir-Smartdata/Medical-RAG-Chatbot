import io
import unittest
from unittest.mock import patch, MagicMock
from django.test import TestCase
from services.pdf_processor import PDFProcessor, PDFProcessingError

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


class TestPDFProcessing(TestCase):
    """
    Automated tests for PDF processing and validation pipeline.
    """

    def setUp(self):
        # Create a simple valid PDF in memory using fitz if available
        self.valid_pdf_stream = io.BytesIO()
        if fitz is not None:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 72), "Patient Diagnosis: Type 2 Diabetes Mellitus (E11.9).\nPrescribed Metformin 500 mg b.i.d.")
            self.valid_pdf_bytes = doc.write()
            doc.close()
        else:
            # Minimal raw PDF bytes for header checks
            self.valid_pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"

    def test_pdf_validation_valid_file(self):
        fake_file = io.BytesIO(self.valid_pdf_bytes)
        fake_file.name = "medical_report.pdf"
        fake_file.size = len(self.valid_pdf_bytes)

        is_valid, msg = PDFProcessor.validate_file(fake_file)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_pdf_validation_invalid_extension(self):
        fake_file = io.BytesIO(b"dummy")
        fake_file.name = "medical_report.txt"
        fake_file.size = 5

        is_valid, msg = PDFProcessor.validate_file(fake_file)
        self.assertFalse(is_valid)
        self.assertIn("Invalid file extension", msg)

    def test_pdf_validation_empty_file(self):
        fake_file = io.BytesIO(b"")
        fake_file.name = "empty.pdf"
        fake_file.size = 0

        is_valid, msg = PDFProcessor.validate_file(fake_file)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg)

    def test_pdf_validation_missing_pdf_header(self):
        fake_file = io.BytesIO(b"CORRUPTED_NOT_A_PDF_CONTENT")
        fake_file.name = "corrupted.pdf"
        fake_file.size = len(fake_file.getvalue())

        is_valid, msg = PDFProcessor.validate_file(fake_file)
        self.assertFalse(is_valid)
        self.assertIn("missing standard PDF header", msg)

    def test_sha256_hash_computation(self):
        fake_file = io.BytesIO(b"Sample Content for Hashing")
        hash1 = PDFProcessor.compute_sha256(fake_file)
        self.assertEqual(len(hash1), 64)

        fake_file2 = io.BytesIO(b"Sample Content for Hashing")
        hash2 = PDFProcessor.compute_sha256(fake_file2)
        self.assertEqual(hash1, hash2)

    @unittest.skipIf(fitz is None, "PyMuPDF not installed in current environment")
    def test_pdf_text_extraction(self):
        pdf_stream = io.BytesIO(self.valid_pdf_bytes)
        result = PDFProcessor.extract_text_and_metadata(pdf_stream)

        self.assertTrue(result["has_text"])
        self.assertEqual(result["total_pages"], 1)
        self.assertIn("Diabetes", result["pages"][0]["text"])
        self.assertIn("Metformin", result["pages"][0]["text"])

    @unittest.skipIf(fitz is None, "PyMuPDF not installed in current environment")
    def test_scanned_pdf_without_text_raises_error(self):
        # Create PDF with an empty page
        doc = fitz.open()
        doc.new_page()
        empty_pdf = doc.write()
        doc.close()

        with self.assertRaises(PDFProcessingError) as ctx:
            PDFProcessor.extract_text_and_metadata(io.BytesIO(empty_pdf))
        self.assertIn("OCR", str(ctx.exception))
