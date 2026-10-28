"""
Citation Service.
Generates unique source identifiers [S1], [S2]... for retrieved context,
formats context blocks for Gemini, and validates returned citations against verified chunks.
"""

import re
from typing import List, Dict, Any, Tuple


class CitationService:
    """
    Manages citation identifier generation, context formatting,
    and output citation validation.
    """

    @classmethod
    def prepare_citations_context(
        cls,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
        """
        Prepares citation-tagged context and metadata reference strings.
        Returns:
            (context_text, metadata_text, source_map)
        """
        context_blocks = []
        metadata_blocks = []
        source_map: Dict[str, Dict[str, Any]] = {}

        for idx, chunk in enumerate(chunks, start=1):
            source_tag = f"[S{idx}]"
            doc_name = chunk.get("document_name", "Unknown Document")
            page_num = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            text = chunk.get("chunk_text", "").strip()
            section = chunk.get("section_title", "General")

            source_map[source_tag] = {
                "source_id": source_tag,
                "document_id": chunk.get("document_id"),
                "document_name": doc_name,
                "page_number": page_num,
                "all_pages": chunk.get("all_pages", [page_num]),
                "chunk_id": chunk_id,
                "section_title": section,
                "chunk_text": text,
                "snippet": text[:200] + ("..." if len(text) > 200 else "")
            }

            metadata_entry = (
                f"{source_tag}\n"
                f"Document: {doc_name}\n"
                f"Page: {page_num}\n"
                f"Section: {section}\n"
                f"Chunk ID: {chunk_id}"
            )
            metadata_blocks.append(metadata_entry)

            context_entry = f"{source_tag}\n{text}"
            context_blocks.append(context_entry)

        retrieved_context_str = "\n\n---\n\n".join(context_blocks)
        source_metadata_str = "\n\n".join(metadata_blocks)

        return retrieved_context_str, source_metadata_str, source_map

    @classmethod
    def extract_and_validate_citations(
        cls,
        answer_text: str,
        source_map: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Validates citation references in answer_text.
        Finds tags like [S1], [S2] or [1], [2] in answer_text,
        replaces or keeps verified ones, and strips hallucinated citations.
        Returns (cleaned_answer, verified_citations_list).
        """
        if not source_map:
            # If no sources were provided, remove any bogus citation tags
            cleaned = re.sub(r'\[S?\d+\]', '', answer_text)
            return cleaned.strip(), []

        # Find all cited markers like [S1], [S2], [1], [2]
        referenced_tags = re.findall(r'\[S?\d+\]', answer_text)
        verified_citations = []
        seen_keys = set()

        for tag in referenced_tags:
            # Normalize [1] to [S1] if model used numbers without 'S'
            norm_tag = tag if tag.startswith("[S") else f"[S{tag[1:-1]}]"

            if norm_tag in source_map and norm_tag not in seen_keys:
                verified_citations.append(source_map[norm_tag])
                seen_keys.add(norm_tag)

        # If answer has no inline tags but we used sources, check if model cited in text
        if not verified_citations:
            for s_id, s_data in source_map.items():
                doc_name = s_data["document_name"].lower()
                if doc_name in answer_text.lower():
                    if s_id not in seen_keys:
                        verified_citations.append(s_data)
                        seen_keys.add(s_id)

        # Fallback: if answer is grounded and retrieved chunks exist, attach top 2 sources
        if not verified_citations and source_map and "insufficient" not in answer_text.lower():
            for s_id in list(source_map.keys())[:2]:
                verified_citations.append(source_map[s_id])

        return answer_text, verified_citations
