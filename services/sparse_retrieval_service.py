"""
Sparse Retrieval Service.
Implements BM25-based keyword retrieval for clinical terms, diagnosis codes,
exact medication names, and numerical dosages.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


class SparseRetrievalService:
    """
    BM25 Keyword-based Sparse Retrieval Engine.
    """

    def __init__(self):
        self.top_k = getattr(settings, 'SPARSE_TOP_K', 20)

    STOPWORDS = {
        'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'can', 'do',
        'for', 'from', 'has', 'have', 'how', 'i', 'in', 'is', 'it', 'its',
        'my', 'no', 'not', 'of', 'on', 'or', 'that', 'the', 'this', 'to',
        'was', 'were', 'what', 'where', 'which', 'who', 'will', 'with', 'you', 'your'
    }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text while preserving clinical numbers, hyphenated terms,
        dosages (e.g., 500mg, 10ml), and diagnosis codes (e.g. E11.9, I10).
        """
        if not text:
            return []
        # Matches alphanumeric tokens including dots, slashes, and hyphens (common in medical nomenclature)
        tokens = re.findall(r'\b[a-zA-Z0-9]+(?:[.\-\/][a-zA-Z0-9]+)*\b', text.lower())
        return tokens

    def search_sparse(
        self,
        query: str,
        corpus_chunks: List[Dict[str, Any]],
        limit: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches chunks using BM25.
        Filters by document_id if provided.
        Returns scored candidates.
        """
        if not query or not corpus_chunks:
            return []

        limit = limit or self.top_k

        # Filter by document_id if specified
        if document_id:
            filtered_chunks = [c for c in corpus_chunks if str(c.get("document_id")) == str(document_id)]
        else:
            filtered_chunks = corpus_chunks

        if not filtered_chunks:
            return []

        all_query_tokens = self.tokenize(query)
        # Prioritize informative keywords over standard stopwords
        query_tokens = [t for t in all_query_tokens if t not in self.STOPWORDS]
        if not query_tokens:
            query_tokens = all_query_tokens

        if not query_tokens:
            return []

        if BM25Okapi is not None:
            tokenized_corpus = [
                self.tokenize(f"{c.get('section_title', '')} {c.get('chunk_text', '')}")
                for c in filtered_chunks
            ]
            tokenized_corpus = [t if t else ["<empty>"] for t in tokenized_corpus]

            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(query_tokens)

            scored_chunks = []
            for idx, raw_score in enumerate(doc_scores):
                doc_toks = tokenized_corpus[idx]
                doc_toks_set = set(doc_toks)
                # Exact or clinical stem/prefix match (e.g. diagnos- for diagnosis/diagnosed)
                match_count = 0
                for q_tok in query_tokens:
                    if q_tok in doc_toks_set:
                        match_count += 1
                    elif len(q_tok) >= 5 and any(d_tok.startswith(q_tok[:5]) for d_tok in doc_toks):
                        match_count += 1

                # If query terms match or raw BM25 score is positive
                if match_count > 0 or raw_score > 0:
                    chunk_item = dict(filtered_chunks[idx])
                    chunk_item["score"] = float(raw_score) + (match_count * 10.0)
                    chunk_item["retrieval_source"] = "sparse"
                    scored_chunks.append(chunk_item)

            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
            return scored_chunks[:limit]

        # Simple TF-IDF/Term Frequency fallback if BM25Okapi is absent
        scored_chunks = []
        for c in filtered_chunks:
            text = c.get("chunk_text", "").lower()
            match_count = sum(text.count(t) for t in query_tokens)
            if match_count > 0:
                item = dict(c)
                item["score"] = float(match_count)
                item["retrieval_source"] = "sparse"
                scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:limit]
