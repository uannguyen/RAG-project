"""
vector_store.py — Thao tác với Qdrant: tạo collection, upsert, search.
"""

import os
import uuid
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

VECTOR_SIZE = 3072  # Google gemini-embedding-001 output size

# Singleton client — tránh tạo connection pool mới mỗi lần gọi
_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    return _client


class VectorStore:
    def __init__(self):
        self.client = _get_client()
        self.collection = os.getenv("QDRANT_COLLECTION", "documents")

    def ensure_collection(self):
        """Tạo collection nếu chưa tồn tại."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"[INFO] Đã tạo collection: {self.collection}")

    def upsert(self, chunks: List[Dict], vectors: List[List[float]]):
        """Lưu chunks + vectors vào Qdrant."""
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "content": chunk["content"],
                    **chunk["metadata"],
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm top_k chunks gần nhất với query vector."""
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "content": r.payload.get("content", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
            }
            for r in results
        ]

    def count(self) -> int:
        """Số lượng chunks đang có trong collection."""
        return self.client.count(collection_name=self.collection).count

    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection)
        print(f"[INFO] Đã xóa collection: {self.collection}")
