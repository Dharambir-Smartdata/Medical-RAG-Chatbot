# Medical AI Assistant — Hybrid RAG Application

An intelligent, production-grade, document-grounded Medical AI Assistant built with **Django**, **Python**, **HTML5/Vanilla CSS/JavaScript**, **Qdrant Vector Database**, and the **Google Gemini API**.

The application ingests medical PDFs (laboratory results, clinical notes, discharge summaries, prescriptions), chunks and indexes content with **Sentence Transformers** into **Qdrant**, retrieves passages using **Hybrid Search (Dense Cosine Similarity + Sparse BM25)** fused via **Reciprocal Rank Fusion (RRF)**, reranks candidates with a **Cross-Encoder**, enforces strict **Medical LLM Guardrails**, and generates answers with **verifiable page-level citations**.

---

## 🏗️ System Architecture

```
User Query
    │
    ▼
Django Web Application
    │
    ├──► Input Guardrails (Length, Prompt Injection & Emergency Triage)
    │
    ├──► Query Embedding (SentenceTransformers: all-MiniLM-L6-v2)
    │
    ├──► Hybrid Retrieval:
    │      ├── Dense Semantic Search (Qdrant Vector DB: Cosine Distance)
    │      └── Sparse Keyword Search (BM25: Medical nomenclature, dosages, ICD codes)
    │
    ├──► Reciprocal Rank Fusion (RRF):
    │      RRF(d) = Σ [ 1 / (k + rank(d)) ]
    │
    ├──► Context Deduplication & Filtering (by User / Document ID)
    │
    ├──► Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    │      Selects Top 5–8 most relevant passages
    │
    ├──► Context Grounding & Citation Tagging ([S1], [S2]...)
    │
    ├──► Google Gemini LLM API (gemini-3.8-flash)
    │
    ├──► Output Guardrails & Citation Verification
    │      Validates grounded claims & removes hallucinated citations
    │
    ▼
Interactive Chat UI (Direct Answer + Clickable Document & Page Citations)
```

---

## 🌟 Key Features

1. **Multi-PDF Document Processing**:
   - Upload multiple clinical documents in a single drag-and-drop operation.
   - Robust PyMuPDF extraction preserving page numbers, section headers, dosages, and terminology.
   - Detects scanned image-only documents and instructs that OCR is required.
   - Duplicate file detection using SHA-256 content hashing.

2. **Configurable Paragraph-Aware Chunking**:
   - Default: ~800 tokens per chunk with ~150 tokens overlap.
   - Respects natural paragraph and sentence boundaries so vital clinical relationships remain intact.
   - Multi-page traceability: records all pages contributing to a chunk.

3. **Dense Vector Embeddings & Qdrant**:
   - Uses **FastEmbed** with `BAAI/bge-small-en-v1.5` (384 dimensions).
   - High-performance ONNX runtime execution with dynamic dimension detection.
   - Stores payloads with `document_id`, `page_number`, `chunk_id`, and `section_title`.
   - Supports both remote Docker Qdrant and local file/in-memory fallback.

4. **Hybrid Retrieval (Dense + Sparse BM25)**:
   - Combines semantic understanding with exact-match BM25 for medical abbreviations, dosages (e.g. `500 mg`, `10 ml`), and ICD codes (e.g. `E11.9`).
   - Merged using Reciprocal Rank Fusion ($k=60$) with deduplication.

5. **Cross-Encoder Reranking**:
   - High-precision reranking using **FastEmbed TextCrossEncoder** (`Xenova/ms-marco-MiniLM-L-6-v2`).
   - Discards irrelevant noise and feeds only the top 5–8 verified chunks to Gemini.

6. **Medical Safety Guardrails**:
   - **Input Guardrails**: Rejects empty prompts, excessive input length (>2000 chars), prompt injection patterns, and routes emergency symptoms (e.g. chest pain, stroke, overdose) to immediate emergency instructions (911).
   - **Context Guardrails**: Clearly responds *"The uploaded documents do not contain sufficient information to answer this question"* if evidence is missing.
   - **Non-Diagnostic Policy**: Never assumes the role of a licensed physician; prevents unauthorized prescribing.
   - **Output Verification**: Verifies citations against actual retrieved chunks; strips unsupported claims.

7. **Verifiable Page-Level Citations**:
   - Every citation card (e.g. `[S1] Medical Report.pdf — Page 5`) is clickable.
   - Opens a source inspector drawer showing the document title, exact page number, clinical section, and highlighted source excerpt.

8. **Minimalistic & Modern SaaS UI**:
   - Pure HTML5, Vanilla CSS, and Vanilla JavaScript with `fetch` API.
   - Responsive sidebar with consultation history, real-time status badges, progressive loading animations, and document scope filtering.

---

## 📋 Technology Stack

- **Backend**: Python 3.10+, Django 5.x, Django REST Framework
- **Frontend**: HTML5, Vanilla CSS (CSS variables, responsive design), Vanilla JavaScript (ES6+ Fetch)
- **Vector Database**: Qdrant (Docker or local disk fallback)
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Sparse Search**: `rank-bm25`
- **Reranker**: Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **LLM**: Google Gemini API (`google-genai` SDK, `gemini-3.8-flash`)

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10 or higher
- Git
- Google Gemini API key ([Get an API key here](https://aistudio.google.com/))
- Docker (optional, for running Qdrant container)

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone <repository_url>
cd "Medical RAG based chatbot"

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Open `.env` and set your credentials:
```ini
DJANGO_SECRET_KEY=your_secret_key_here
DEBUG=True
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash
QDRANT_URL=http://localhost:6333
CHUNK_SIZE=800
CHUNK_OVERLAP=150
```

### 4. (Optional) Start Qdrant with Docker
If Docker is installed:
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```
> *Note: If Docker is not running, the application automatically uses local disk Qdrant storage at `./qdrant_storage`.*

### 5. Run Migrations & Start Django
```bash
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```
Open your browser and navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🐳 Running with Docker Compose

To run the complete stack (Django + Qdrant container) via Docker Compose:
```bash
docker-compose up --build
```
Access the application at: **[http://localhost:8000](http://localhost:8000)**

---

## 🧪 Running Automated Tests

Run the full automated test suite covering PDF extraction, chunking, dense/sparse retrieval, RRF, reranking, medical guardrails, and Django views:
```bash
python manage.py test tests
```

---

## 📖 API Endpoints Reference

### Document Management
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/documents/` | List all uploaded documents and processing statuses |
| `POST` | `/documents/upload/` | Upload multiple medical PDFs (`multipart/form-data`) |
| `GET` | `/documents/<uuid:document_id>/` | Retrieve document details and extracted chunks |
| `DELETE` | `/documents/<uuid:document_id>/delete/` | Delete document, DB chunks, and Qdrant vector points |

### Chat & Consultations
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main Chat User Interface |
| `POST` | `/chat/` | Submit a medical question to the RAG pipeline |
| `GET` | `/chat/history/` | List user chat sessions |
| `POST` | `/chat/new/` | Create a new consultation session |
| `GET` | `/chat/session/<uuid:session_id>/` | Retrieve message transcript and verified citations |
| `DELETE` | `/chat/session/<uuid:session_id>/delete/` | Delete consultation history |

---

## 🔒 Security, Privacy & Medical Limitations

1. **Document & Session Isolation**: Queries and document chunks are filtered by user/session. Users cannot view or query documents belonging to others.
2. **Untrusted Input Handling**: Uploaded PDF text is treated as untrusted data. Embedded instructions attempting to override system prompts are ignored.
3. **Emergency Routing**: The system detects critical symptoms (e.g. chest pain, stroke, overdose) and displays immediate directives to contact emergency services (911).
4. **Non-Diagnostic Informational Assistant**: This software provides document-grounded informational summaries and cannot replace professional medical diagnosis, advice, or treatment.
5. **No Medical Content in Server Logs**: Clinical passages and API keys are strictly excluded from logging outputs.

---

## 🛠️ Troubleshooting

- **Gemini API Key Missing**:
  If `GEMINI_API_KEY` is not set, the assistant operates in local synthesis mode extracting verified sentences from retrieved chunks. Set `GEMINI_API_KEY` in `.env` for generative answers.
- **Scanned PDF Text Extraction**:
  If an uploaded PDF is an image scan without an embedded text layer, the system flags: *"Unable to extract text from this PDF. OCR is required for scanned documents."*
- **Qdrant Connection**:
  If the remote Qdrant Docker container is unreachable, the system automatically falls back to local file-based Qdrant storage (`./qdrant_storage`), ensuring zero downtime.
