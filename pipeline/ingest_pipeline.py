"""
ingest_pipeline.py — Pipeline hoàn chỉnh: load → chunk → embed → lưu vào Qdrant.
"""

import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

from ingestion.loader import load_directory, load_file
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_texts
from retrieval.vector_store import VectorStore


def ingest_directory(dir_path: str, batch_size: int = 32):
    """Ingest toàn bộ documents trong thư mục vào Qdrant."""
    store = VectorStore()
    store.ensure_collection()

    # FIXED: safe int parsing with fallback to prevent ValueError crash
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", 512))
    except ValueError:
        chunk_size = 512
    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    except ValueError:
        chunk_overlap = 50

    total_chunks = 0
    print(f"[INFO] Đang load documents từ: {dir_path}")

    for doc in load_directory(dir_path):
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        doc_chunk_count = len(chunks)
        print(f"  ✓ {doc['metadata']['filename']} → {doc_chunk_count} chunks")

        # Xử lý từng document ngay, không tích lũy tất cả vào RAM
        for i in range(0, doc_chunk_count, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["content"] for c in batch]
            vectors = embed_texts(texts)
            store.upsert(batch, vectors)

        total_chunks += doc_chunk_count

    print(f"\n[DONE] Đã ingest {total_chunks} chunks vào Qdrant.")


def ingest_file(file_path: str, batch_size: int = 8):
    """Ingest một file đơn lẻ, xử lý theo batch nhỏ để tránh tràn RAM."""
    store = VectorStore()
    store.ensure_collection()

    # FIXED: safe int parsing with fallback to prevent ValueError crash
    try:
        chunk_size = int(os.getenv("CHUNK_SIZE", 512))
    except ValueError:
        chunk_size = 512
    try:
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    except ValueError:
        chunk_overlap = 50

    doc = load_file(file_path)
    chunks = chunk_document(doc, chunk_size, chunk_overlap)
    total = len(chunks)
    print(f"[INFO] {doc['metadata']['filename']} → {total} chunks")

    # Xử lý từng batch nhỏ thay vì load tất cả vào RAM cùng lúc
    for i in tqdm(range(0, total, batch_size), desc="Embedding"):
        batch = chunks[i : i + batch_size]
        texts = [c["content"] for c in batch]
        vectors = embed_texts(texts)
        store.upsert(batch, vectors)

    print(f"[DONE] Đã ingest {total} chunks.")
