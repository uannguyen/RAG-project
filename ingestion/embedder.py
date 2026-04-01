"""
embedder.py — Tạo vector embeddings dùng Google Embedding API (free, không tốn RAM).
"""

import os
import time
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/gemini-embedding-001"


def embed_texts(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """Chuyển list text thành list vectors qua Google API."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="RETRIEVAL_DOCUMENT",
        )
        all_vectors.extend(result["embedding"])
        # Rate limit: tránh bị block khi ingest nhiều
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return all_vectors


def embed_query(query: str) -> List[float]:
    """Embed một câu query duy nhất."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]
