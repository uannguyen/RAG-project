"""
server.py — FastAPI server: upload documents, hỏi đáp.
"""

import os
import asyncio
import logging
import uuid as _uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # FIXED: 50 MB upload limit to prevent disk exhaustion

load_dotenv()

from pipeline.ingest_pipeline import ingest_file
from pipeline.rag_pipeline import ask
from retrieval.vector_store import VectorStore

app = FastAPI(title="RAG API", version="1.0.0")

# FIXED: add CORS middleware so frontend on different origin can call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Models ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[str]


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    store = VectorStore()
    try:
        count = store.count()
    except Exception as e:
        logger.exception("Failed to get stats from Qdrant")  # FIXED: log instead of silently swallowing
        count = 0
    return {"total_chunks": count, "collection": store.collection}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload và ingest một file (PDF, DOCX, TXT)."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md"}
    safe_name = Path(file.filename).name  # FIXED: strip directory components to prevent path traversal
    ext = Path(safe_name).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Không hỗ trợ định dạng: {ext}")

    # FIXED: unique filename to prevent race condition on concurrent uploads
    unique_name = f"{_uuid.uuid4().hex}_{safe_name}"
    dest = UPLOAD_DIR / unique_name

    # FIXED: enforce max upload size to prevent disk exhaustion
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_UPLOAD_SIZE:
                    f.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File quá lớn (tối đa 50 MB).")
                f.write(chunk)
    finally:
        await file.close()

    try:
        await asyncio.to_thread(ingest_file, str(dest))
    except Exception as e:
        logger.exception("Ingestion failed for %s", safe_name)  # FIXED: log real error server-side
        raise HTTPException(status_code=500, detail="Lỗi khi xử lý file. Vui lòng thử lại.")

    return {"message": f"Đã ingest: {safe_name}"}


@app.post("/ask", response_model=QuestionResponse)
def ask_question(req: QuestionRequest):
    """Hỏi đáp dựa trên tài liệu đã upload."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")
    try:
        result = ask(req.question)
    except ResourceExhausted as e:
        raise HTTPException(status_code=429, detail=str(e))
    return QuestionResponse(answer=result["answer"], sources=result["sources"])


# ── Static UI ────────────────────────────────────────────────────────────────

UI_DIR = Path(__file__).parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(UI_DIR / "index.html"))
