"""
chunker.py — Chia văn bản thành các chunks nhỏ để embedding.
"""

from typing import List


def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Chia text thành chunks theo ký tự, ưu tiên cắt tại dấu xuống dòng hoặc dấu chấm.
    """
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Ưu tiên cắt tại newline hoặc dấu chấm gần nhất
        cut = end
        for sep in ["\n\n", "\n", ". ", "。", " "]:
            idx = text.rfind(sep, start, end)
            if idx != -1 and idx > start:
                cut = idx + len(sep)
                break

        chunk = text[start:cut].strip()
        if chunk:
            chunks.append(chunk)

        # Đảm bảo start luôn tiến về phía trước ít nhất 1 ký tự
        new_start = cut - chunk_overlap
        start = max(new_start, start + 1)

    return chunks


def chunk_document(doc: dict, chunk_size: int = 512, chunk_overlap: int = 50) -> List[dict]:
    """
    Nhận dict {content, metadata} và trả về list các chunks với metadata.
    """
    chunks = split_text(doc["content"], chunk_size, chunk_overlap)
    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "content": chunk,
            "metadata": {
                **doc["metadata"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        })
    return result
