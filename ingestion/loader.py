"""
loader.py — Load PDF, Word, và Text files thành plain text.
"""

import os
from pathlib import Path
from typing import Generator


def load_pdf(file_path: str) -> str:
    import pdfplumber
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def load_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".doc": load_docx,
    ".txt": load_txt,
    ".md": load_txt,
}


def load_file(file_path: str) -> dict:
    """Load một file, trả về dict gồm content và metadata."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")
    content = LOADERS[ext](file_path)
    return {
        "content": content,
        "metadata": {
            "source": str(path),
            "filename": path.name,
            "extension": ext,
        },
    }


def load_directory(dir_path: str) -> Generator[dict, None, None]:
    """Load tất cả documents trong một thư mục (đệ quy)."""
    for root, _, files in os.walk(dir_path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in LOADERS:
                full_path = os.path.join(root, fname)
                try:
                    yield load_file(full_path)
                except Exception as e:
                    print(f"[WARN] Bỏ qua {full_path}: {e}")
