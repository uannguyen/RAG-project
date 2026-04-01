"""
server.py — FastAPI server: upload documents, hỏi đáp.
"""

import os
import shutil
import tempfile
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

from pipeline.ingest_pipeline import ingest_file
from pipeline.rag_pipeline import ask
from retrieval.vector_store import VectorStore

app = FastAPI(title="RAG API", version="1.0.0")

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
    except Exception:
        count = 0
    return {"total_chunks": count, "collection": store.collection}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload và ingest một file (PDF, DOCX, TXT)."""
    allowed = {".pdf", ".docx", ".doc", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Không hỗ trợ định dạng: {ext}")

    dest = UPLOAD_DIR / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    try:
        await asyncio.to_thread(ingest_file, str(dest))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": f"Đã ingest: {file.filename}"}


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
