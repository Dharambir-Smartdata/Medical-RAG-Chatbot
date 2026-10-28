"""
PDF Processing Service.
Handles PDF validation, PyMuPDF text extraction, page-level mapping, and metadata extraction.
"""

import os
import hashlib
from typing import List, Dict, Any, Tuple
from django.conf import settings

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """
    Robust PDF processor supporting multi-page text extraction,
    security checks, validation, and metadata extraction.
    """

    MAX_FILE_SIZE_BYTES = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 20) * 1024 * 1024

    @classmethod
    def validate_file(cls, file_obj) -> Tuple[bool, str]:
        """
        Validates file extension, size, MIME/magic header, and basic integrity.
        Returns (is_valid, error_message).
        """
        filename = getattr(file_obj, 'name', '')
        if not filename.lower().endswith('.pdf'):
            return False, "Invalid file extension. Only PDF (.pdf) files are allowed."

        # File size check
        size = getattr(file_obj, 'size', 0)
        if size == 0:
            return False, "The uploaded file is empty (0 bytes)."
        if size > cls.MAX_FILE_SIZE_BYTES:
            max_mb = getattr(settings, 'MAX_UPLOAD_SIZE_MB', 20)
            return False, f"File size exceeds the maximum limit of {max_mb} MB."

        # Header check for %PDF
        file_obj.seek(0)
        header = file_obj.read(5)
        file_obj.seek(0)
        if not header.startswith(b'%PDF-'):
            return False, "File is not a valid PDF document (missing standard PDF header)."

        return True, ""

    @classmethod
    def compute_sha256(cls, file_obj) -> str:
        """Computes SHA-256 hash of the file for duplicate detection."""
        hasher = hashlib.sha256()
        file_obj.seek(0)
        for chunk in file_obj.chunks() if hasattr(file_obj, 'chunks') else [file_obj.read()]:
            hasher.update(chunk)
        file_obj.seek(0)
        return hasher.hexdigest()

    @classmethod
    def extract_text_and_metadata(cls, file_path_or_stream) -> Dict[str, Any]:
        """
        Extracts text page by page with PyMuPDF.
        Returns:
            {
                "pages": [{"page_number": int, "text": str}],
                "total_pages": int,
                "metadata": dict,
                "has_text": bool
            }
        """
        if fitz is None:
            raise PDFProcessingError("PyMuPDF (fitz) is not installed in the environment.")

        doc = None
        try:
            if isinstance(file_path_or_stream, str):
                if not os.path.exists(file_path_or_stream):
                    raise PDFProcessingError(f"File not found: {file_path_or_stream}")
                doc = fitz.open(file_path_or_stream)
            elif hasattr(file_path_or_stream, 'read'):
                file_path_or_stream.seek(0)
                content = file_path_or_stream.read()
                doc = fitz.open(stream=content, filetype="pdf")
            else:
                doc = fitz.open(file_path_or_stream)

            # Check encryption/password
            if doc.is_encrypted or doc.needs_pass:
                raise PDFProcessingError(
                    "This PDF document is password-protected or encrypted. Please provide an unprotected PDF."
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise PDFProcessingError("PDF contains no pages.")

            pages_data: List[Dict[str, Any]] = []
            total_chars = 0

            for page_idx in range(total_pages):
                page = doc[page_idx]
                text = page.get_text("text") or ""
                # Normalize linebreaks and strip whitespace
                clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
                char_count = len(clean_text)
                total_chars += char_count

                pages_data.append({
                    "page_number": page_idx + 1,  # 1-indexed
                    "text": clean_text,
                    "char_count": char_count
                })

            # Check if any extractable text was found
            if total_chars == 0:
                raise PDFProcessingError(
                    "Unable to extract text from this PDF. The document may be scanned or image-only. "
                    "OCR (Optical Character Recognition) is required for scanned documents."
                )

            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "format": doc.metadata.get("format", ""),
            }

            return {
                "pages": pages_data,
                "total_pages": total_pages,
                "metadata": metadata,
                "has_text": True,
                "total_characters": total_chars
            }

        except Exception as e:
            if isinstance(e, PDFProcessingError):
                raise
            raise PDFProcessingError(f"Corrupted or unreadable PDF: {str(e)}") from e
        finally:
            if doc:
                try:
                    doc.close()
                except Exception:
                    pass
