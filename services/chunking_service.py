"""
Chunking Module.
Implements configurable, paragraph-aware, recursive text chunking
preserving medical terms, dosages, ICD codes, and page-level source traceability.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from django.conf import settings


class ChunkingService:
    """
    Configurable document chunker with paragraph and sentence boundary preservation.
    """

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or getattr(settings, 'CHUNK_SIZE', 800)
        self.chunk_overlap = chunk_overlap or getattr(settings, 'CHUNK_OVERLAP', 150)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate based on words and punctuation (approx 1 word ~ 1.3 tokens)."""
        words = len(text.split())
        return int(words * 1.3)

    @staticmethod
    def _detect_section_title(text: str) -> str:
        """
        Attempts to identify clinical or standard section headings
        like 'Diagnosis', 'Chief Complaint', 'Medications', 'Lab Results', etc.
        """
        clinical_sections = [
            "chief complaint", "history of present illness", "past medical history",
            "medications", "allergies", "physical examination", "assessment and plan",
            "diagnosis", "laboratory findings", "diagnostic imaging", "discharge summary",
            "treatment plan", "vital signs", "clinical impression", "progress note"
        ]
        first_lines = text.strip().split("\n")[:3]
        for line in first_lines:
            clean = line.strip().rstrip(":").lower()
            for sec in clinical_sections:
                if sec in clean or clean.startswith(sec):
                    return line.strip()
        # Fallback: check if the first line is short and uppercase/title-like
        if first_lines and len(first_lines[0].split()) <= 6 and len(first_lines[0]) <= 50:
            return first_lines[0].strip()
        return ""

    def chunk_document(
        self,
        document_id: str,
        document_name: str,
        pages_data: List[Dict[str, Any]],
        upload_timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes page-level text items and generates overlapping chunks preserving paragraph
        boundaries and tracking origin page numbers.
        """
        ts = upload_timestamp or datetime.utcnow().isoformat()
        chunks: List[Dict[str, Any]] = []

        # Prepare list of paragraph units tagged with their page number
        paragraph_units = []
        for p in pages_data:
            page_num = p["page_number"]
            raw_text = p.get("text", "")
            # Split by double newline or distinct paragraphs
            paras = [para.strip() for para in re.split(r'\n\s*\n', raw_text) if para.strip()]
            if not paras and raw_text.strip():
                paras = [raw_text.strip()]

            for para in paras:
                paragraph_units.append({
                    "text": para,
                    "page_number": page_num
                })

        if not paragraph_units:
            return []

        current_paras = []
        current_token_count = 0
        chunk_idx = 0

        unit_idx = 0
        while unit_idx < len(paragraph_units):
            unit = paragraph_units[unit_idx]
            unit_tokens = self._estimate_tokens(unit["text"])

            # If a single paragraph is larger than chunk_size, split by sentences
            if unit_tokens > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', unit["text"])
                for s in sentences:
                    s_tokens = self._estimate_tokens(s)
                    if current_token_count + s_tokens > self.chunk_size and current_paras:
                        chunk_record = self._create_chunk_record(
                            document_id=document_id,
                            document_name=document_name,
                            chunk_index=chunk_idx,
                            items=current_paras,
                            upload_timestamp=ts
                        )
                        chunks.append(chunk_record)
                        chunk_idx += 1
                        # Retain overlap from end of current_paras
                        current_paras, current_token_count = self._get_overlap_items(current_paras)

                    current_paras.append({"text": s, "page_number": unit["page_number"]})
                    current_token_count += s_tokens

                unit_idx += 1
                continue

            if current_token_count + unit_tokens > self.chunk_size and current_paras:
                chunk_record = self._create_chunk_record(
                    document_id=document_id,
                    document_name=document_name,
                    chunk_index=chunk_idx,
                    items=current_paras,
                    upload_timestamp=ts
                )
                chunks.append(chunk_record)
                chunk_idx += 1
                current_paras, current_token_count = self._get_overlap_items(current_paras)

            current_paras.append(unit)
            current_token_count += unit_tokens
            unit_idx += 1

        # Final leftover chunk
        if current_paras:
            chunk_record = self._create_chunk_record(
                document_id=document_id,
                document_name=document_name,
                chunk_index=chunk_idx,
                items=current_paras,
                upload_timestamp=ts
            )
            chunks.append(chunk_record)

        return chunks

    def _get_overlap_items(self, items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Collects trailing items up to chunk_overlap tokens to maintain continuity."""
        overlap_items = []
        token_sum = 0
        for it in reversed(items):
            it_tokens = self._estimate_tokens(it["text"])
            if token_sum + it_tokens <= self.chunk_overlap or not overlap_items:
                overlap_items.insert(0, it)
                token_sum += it_tokens
            else:
                break
        return overlap_items, token_sum

    def _create_chunk_record(
        self,
        document_id: str,
        document_name: str,
        chunk_index: int,
        items: List[Dict[str, Any]],
        upload_timestamp: str
    ) -> Dict[str, Any]:
        """Constructs standardized chunk metadata dictionary."""
        combined_text = "\n\n".join(item["text"] for item in items).strip()
        pages = sorted(list({item["page_number"] for item in items}))
        primary_page = pages[0] if pages else 1
        section_title = self._detect_section_title(combined_text)

        chunk_id = f"{document_id}_chunk_{chunk_index:04d}"

        return {
            "chunk_id": chunk_id,
            "document_id": str(document_id),
            "document_name": document_name,
            "page_number": primary_page,
            "all_pages": pages,
            "chunk_index": chunk_index,
            "chunk_text": combined_text,
            "section_title": section_title or "General Section",
            "token_estimate": self._estimate_tokens(combined_text),
            "uploaded_at": upload_timestamp
        }
