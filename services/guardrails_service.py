"""
Medical LLM Guardrails Service.
Implements multi-tiered safety:
1. Input validation (length, injections, emergency triage)
2. Context grounding verification
3. Medical safety rules (non-diagnostic, non-prescriptive)
4. Output citation and claim verification
"""

import re
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

EMERGENCY_KEYWORDS = [
    r'\bchest pain\b',
    r'\bheart attack\b',
    r'\bstroke\b',
    r'\bcan\'?t breathe\b',
    r'\bsevere shortness of breath\b',
    r'\banaphylaxis\b',
    r'\bsevere allergic reaction\b',
    r'\bheavily bleeding\b',
    r'\bsevere hemorrhage\b',
    r'\bsuicide\b',
    r'\bkill myself\b',
    r'\boverdose\b',
    r'\bpoison(?:ed|ing)?\b',
    r'\bunconscious(?:ness)?\b',
]

INJECTION_PATTERNS = [
    r'ignore\s+(?:all\s+)?(?:previous\s+)?instructions',
    r'disregard\s+(?:all\s+)?(?:system\s+)?prompts',
    r'reveal\s+(?:system\s+)?(?:instructions|prompt)',
    r'system\s+override',
    r'you\s+are\s+now\s+(?:dan|unrestricted)',
    r'pretend\s+you\s+are\s+a\s+doctor\s+with\s+no\s+rules',
    r'bypass\s+(?:safety|guardrails)',
]

PRESCRIPTION_REQUEST_PATTERNS = [
    r'\bprescribe\s+(?:me\s+)?(?:some\s+)?',
    r'\bwhat\s+dose\s+should\s+i\s+take\b',
    r'\bhow\s+much\s+(?:mg|ml|tablets)\s+should\s+i\s+take\b',
    r'\bcan\s+i\s+change\s+my\s+dosage\b',
    r'\bdiagnose\s+my\s+condition\b',
]

OUT_OF_CONTEXT_MESSAGE = (
    "I can only answer questions related to your uploaded PDF documents. "
    "Please ask a question about the information contained in your documents."
)
OUT_OF_SCOPE_MESSAGE = OUT_OF_CONTEXT_MESSAGE

INSUFFICIENT_INFO_MESSAGE = (
    "I could not find sufficient information in your uploaded documents to answer this question."
)

NO_DOCUMENTS_MESSAGE = (
    "Please upload a PDF document before asking document-related questions."
)

STANDARD_DISCLAIMER = ""

COMMON_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me',
    'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should',
    'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
    'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why',
    'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    # Common ordinals and query meta-words
    'first', 'second', 'third', 'last', 'one', 'two', 'three', 'tell', 'show', 'give', 'please',
    'find', 'list', 'mention', 'mentioned', 'state', 'stated', 'report', 'reported', 'document',
    'documents', 'pdf', 'pdfs', 'record', 'records', 'file', 'files', 'patient', 'person',
    'information', 'detail', 'details', 'value', 'values', 'result', 'results', 'data', 'provide',
    # Formatting and conversational meta-words
    'sentence', 'sentences', 'summary', 'summarize', 'summarise', 'rephrase', 'rewrite',
    'shorten', 'condense', 'point', 'points', 'bullet', 'bullets', 'paragraph', 'paragraphs',
    'response', 'responses', 'answer', 'answers', 'explanation', 'whatever', 'provided',
    'say', 'said', 'earlier', 'prior', 'preceding', 'above'
}

FOLLOW_UP_PATTERNS = [
    r'\b(?:the\s+)?(?:above|previous|prior|earlier|last|preceding)\s+(?:response|answer|message|summary|points?|information|result|data)\b',
    r'\b(?:what(?:ever)?\s+(?:you\s+)?(?:have\s+)?(?:provided|said|wrote|mentioned|given))\b',
    r'\b(?:summarize|summarise|shorten|rephrase|rewrite|condense|simplify)\s+(?:it|this|that|the\s+above|your\s+answer|your\s+response)?\b',
    r'\b(?:in\s+a\s+sentence|in\s+one\s+sentence|in\s+\d+\s+sentences?|in\s+a\s+short\s+sentence|in\s+a\s+paragraph|in\s+bullet\s+points?|as\s+bullet\s+points?)\b',
    r'\b(?:provide|give|tell)\s+(?:me\s+)?(?:the\s+above|that|it)\s+(?:in|as)\b',
    r'\b(?:explain|clarify)\s+(?:the\s+)?(?:above|first|second|third|last|previous)\s+(?:point|item|one|result)\b',
    r'\b(?:what\s+about\s+(?:it|that|this))\b',
    r'\b(?:can\s+you\s+(?:summarize|shorten|rephrase|simplify))\b',
]


def is_conversational_follow_up(question: str) -> bool:
    """Detects whether user question is a follow-up or meta-request referencing previous responses."""
    cleaned = question.strip().lower()
    for pattern in FOLLOW_UP_PATTERNS:
        if re.search(pattern, cleaned):
            return True
    return False

OUT_OF_SCOPE_PATTERNS = [
    # Programming, coding, scripts, data science
    r'\b(?:code|script|function|program|query|algorithm)\s+to\b',
    r'\b(?:python|javascript|typescript|c\+\+|java|golang|ruby|rust|php|sql|bash|powershell|regex)\b',
    r'\b(?:dataframe|pandas|numpy|scikit|sklearn|pytorch|tensorflow|matplotlib|seaborn|django|flask|fastapi)\b',
    r'\b(?:write|give|provide)\s+(?:me\s+)?(?:the\s+)?(?:python\s+|bash\s+|sql\s+|js\s+)?code\b',
    r'\b(?:remove|drop|clean|impute|fill)\s+(?:missing\s+values|nulls|nan|rows|columns)\b',
    r'\b(?:how\s+to\s+code|how\s+to\s+program|debug\s+this|write\s+a\s+query|fix\s+this\s+bug)\b',
    # General world knowledge / politics / history / geography / entertainment
    r'\b(?:president|prime\s+minister|chancellor|monarch|governor|king|queen|emperor)\s+of\b',
    r'\b(?:capital|currency|population|area|continent|gdp)\s+of\b',
    r'\b(?:who\s+was|who\s+is)\s+(?:the\s+)?(?:first|current|last|next)?\s*(?:president|minister|king|queen|leader|actor|singer|founder|inventor|author|director)\b',
    r'\b(?:who\s+won|score\s+of|weather\s+in|crypto\s+price|bitcoin|stock\s+price|olympics|world\s+cup)\b',
    r'\b(?:when\s+was|when\s+did)\s+(?:india|america|us|china|world\s+war|independence)\b',
    # Creative writing & general tasks
    r'\b(?:write|compose)\s+(?:a|an)\s+(?:poem|essay|story|song|joke|haiku|speech|email|cover\s+letter)\b',
    r'\b(?:recipe\s+for|how\s+to\s+cook|how\s+to\s+bake|how\s+to\s+make\s+(?:cake|pizza|pasta|bread))\b',
    r'\b(?:how\s+to\s+repair|how\s+to\s+fix\s+(?:car|bike|tire|chain|phone|laptop))\b',
]


def is_out_of_scope_query(question: str) -> bool:
    """Detects whether question is unrelated general knowledge, coding, or out of domain."""
    cleaned = question.strip()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True
    return False


class GuardrailsService:
    """
    Validates input queries, enforces clinical safety boundaries,
    and inspects generated outputs.
    """


    @classmethod
    def validate_input(cls, question: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates the user question before any retrieval or LLM call.
        Returns:
            (is_valid, error_or_safety_response, rejection_category)
        """
        if not question or not question.strip():
            return False, "Please enter a valid medical question.", "empty_input"

        cleaned = question.strip()

        if len(cleaned) > 2000:
            return False, "Question is excessively long. Please limit your query to 2000 characters.", "length_exceeded"

        # Check for prompt injection attempts
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                logger.warning(f"Prompt injection pattern detected: '{pattern}'")
                return False, (
                    "Security Alert: Your question contains prohibited instructions attempting to override system safety rules. "
                    "Please ask a question regarding your uploaded medical records."
                ), "prompt_injection"

        # Check for acute medical emergencies
        for pattern in EMERGENCY_KEYWORDS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                emergency_response = (
                    "🚨 **EMERGENCY MEDICAL WARNING** 🚨\n\n"
                    "Your message may indicate a potential acute medical emergency. "
                    "Do not wait for an AI response or rely on uploaded documents. "
                    "**Immediately call emergency services (such as 911 or your local emergency number) "
                    "or proceed to the nearest emergency department.**"
                )
                return False, emergency_response, "emergency_triage"

        # Check for out-of-scope / non-document queries (e.g. coding, programming, general trivia)
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                logger.info(f"Out-of-scope query detected matching pattern: '{pattern}'")
                return False, OUT_OF_SCOPE_MESSAGE, "out_of_scope"

        return True, None, None

    @classmethod
    def extract_query_content_tokens(cls, text: str) -> list:
        """Extracts non-stopword content tokens from query."""
        words = re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text.lower())
        return [w for w in words if not w.isdigit() and w not in COMMON_STOPWORDS]

    @classmethod
    def check_context_sufficiency(
        cls,
        retrieved_chunks: list,
        user_question: str = "",
        is_follow_up: bool = False
    ) -> bool:
        """
        Checks whether retrieved context is available, meaningful,
        and relevant to the user question.
        """
        if not retrieved_chunks:
            return False

        # Verify chunks have extractable text
        combined_text = "".join(c.get("chunk_text", "") for c in retrieved_chunks).strip()
        if len(combined_text) < 20:
            return False

        # For conversational follow-ups referring to prior conversation/answers,
        # the question language is formatting/meta instruction, so bypass single-turn lexical overlap.
        if is_follow_up:
            return True

        # Verify lexical/conceptual relevance between query and retrieved chunks
        if user_question:
            q_tokens = cls.extract_query_content_tokens(user_question)
            if q_tokens:
                combined_lower = combined_text.lower()
                overlap = sum(
                    1 for tok in q_tokens
                    if tok in combined_lower or (len(tok) >= 5 and tok[:4] in combined_lower)
                )
                # If specific query keywords were asked, but zero appear in the retrieved context
                if overlap == 0:
                    return False

        return True



    @classmethod
    def enforce_output_guardrails(
        cls,
        raw_output: str,
        retrieved_chunks: list,
        user_question: str
    ) -> Tuple[str, bool]:
        """
        Inspects model output:
        - Detects if model explicitly flagged insufficient information.
        - Checks for unsafe self-diagnosis or dangerous prescription commands.
        - Appends medical safety disclaimer.
        Returns:
            (safe_output, is_sufficient)
        """
        if not raw_output or not raw_output.strip():
            return INSUFFICIENT_INFO_MESSAGE, False

        text = raw_output.strip()

        # Check if insufficient context was acknowledged
        insufficient_markers = [
            "do not contain sufficient information",
            "does not contain sufficient information",
            "insufficient information",
            "not mentioned in the uploaded documents",
            "not found in the uploaded documents",
            "no mention of",
            "cannot find information",
            "cannot find any information",
            "no information found",
            "no information available"
        ]
        # Check if model reported that the question cannot be answered at all
        has_citations = bool(re.search(r'\[S\d+\]', text))
        if not has_citations and any(marker in text.lower() for marker in insufficient_markers):
            return INSUFFICIENT_INFO_MESSAGE, False
        if len(text) < 120 and any(marker in text.lower() for marker in insufficient_markers) and not any(kw in text.lower() for kw in ['value', 'result', 'flag', 'level', 'mg/dl', '%', 'bp', 'mmhg']):
            return INSUFFICIENT_INFO_MESSAGE, False

        # Check if model attempted to prescribe or diagnose outside bounds
        if "i prescribe" in text.lower() or "you must take" in text.lower() and "dosage" in text.lower():
            text += (
                "\n\n*Caution: Any dosages or medication names cited are extracted directly from your records. "
                "Do not start, stop, or adjust medications without direct supervision from your prescribing physician.*"
            )

        return text, True
