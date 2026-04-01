# RAG Project

A Retrieval-Augmented Generation (RAG) system that lets you upload documents and ask questions about them in natural language. Built with FastAPI, Qdrant, and Google Gemini.

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Vector DB:** Qdrant (via Docker)
- **Embeddings:** Google Gemini Embedding (`gemini-embedding-001`, 3072 dims)
- **LLM:** Google Gemini (configurable via `.env`, default `gemini-2.5-flash`)
- **Document parsing:** pdfplumber (PDF), python-docx (DOCX)

## Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Google Gemini API key — lấy tại [https://aistudio.google.com](https://aistudio.google.com)

## Setup

```bash
# 1. Clone repo
git clone git@github.com:uannguyen/RAG-project.git
cd RAG-project

# 2. Tạo virtualenv và cài dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Cấu hình environment
cp .env.example .env
# Điền GEMINI_API_KEY vào .env

# 4. Khởi động Qdrant
docker compose up -d

# 5. Chạy server
python3 run.py
```

Mở trình duyệt tại [http://localhost:8000](http://localhost:8000)

## Usage

### Web UI
Truy cập `http://localhost:8000` để dùng giao diện kéo-thả upload file và chat.

### API

```bash
# Upload file
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Hỏi đáp
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Nội dung chính của tài liệu là gì?"}'

# Kiểm tra số chunks đã ingest
curl http://localhost:8000/stats
```

### CLI Ingest

```bash
# Ingest một file
python3 scripts/ingest.py --file data/documents/file.pdf

# Ingest toàn bộ thư mục
python3 scripts/ingest.py --dir data/documents
```

## Supported File Formats

`.pdf` · `.docx` · `.doc` · `.txt` · `.md`

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model dùng để sinh câu trả lời |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Tên collection |
| `CHUNK_SIZE` | `512` | Số ký tự mỗi chunk |
| `CHUNK_OVERLAP` | `50` | Overlap giữa các chunk |
| `TOP_K` | `5` | Số chunks truy xuất mỗi query |
