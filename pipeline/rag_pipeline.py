"""
rag_pipeline.py — Pipeline hỏi-đáp: embed query → retrieve → generate.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from ingestion.embedder import embed_query
from retrieval.vector_store import VectorStore
from generation.llm import generate_answer


def ask(question: str) -> dict:
    """
    Hỏi một câu và nhận câu trả lời từ tài liệu.
    Trả về dict gồm answer và sources.
    """
    # FIXED: safe int parsing with fallback to prevent ValueError crash
    try:
        top_k = int(os.getenv("TOP_K", 5))
    except ValueError:
        top_k = 5

    # 1. Embed câu hỏi
    query_vector = embed_query(question)

    # 2. Tìm chunks liên quan
    store = VectorStore()
    chunks = store.search(query_vector, top_k=top_k)

    if not chunks:
        return {
            "answer": "Không tìm thấy tài liệu liên quan.",
            "sources": [],
        }

    # 3. Sinh câu trả lời
    answer = generate_answer(question, chunks)

    sources = list({c["metadata"].get("filename", "unknown") for c in chunks})

    return {
        "answer": answer,
        "sources": sources,
    }  # FIXED: removed unused "chunks" key not in response model
