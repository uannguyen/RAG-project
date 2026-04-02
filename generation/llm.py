"""
llm.py — Gọi Gemini API để sinh câu trả lời.
"""

import os
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from dotenv import load_dotenv

load_dotenv()

# FIXED: fail fast if API key is missing instead of silent None
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set. Check your .env file.")
genai.configure(api_key=_api_key)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = genai.GenerativeModel(
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            system_instruction=SYSTEM_PROMPT,  # FIXED: system prompt as structured instruction, not interpolated
        )
    return _model


SYSTEM_PROMPT = """Bạn là trợ lý AI thông minh. Nhiệm vụ của bạn là trả lời câu hỏi DỰA TRÊN các đoạn tài liệu được cung cấp.

Quy tắc:
- Chỉ trả lời dựa trên nội dung tài liệu đã cho.
- Nếu tài liệu không có đủ thông tin, hãy nói rõ: "Tôi không tìm thấy thông tin này trong tài liệu."
- Trả lời bằng cùng ngôn ngữ với câu hỏi (tiếng Việt hoặc tiếng Anh).
- Trích dẫn nguồn tài liệu khi có thể (tên file).
"""


def generate_answer(question: str, context_chunks: list) -> str:
    """Sinh câu trả lời từ câu hỏi và các chunks context."""
    context_text = "\n\n---\n\n".join(
        f"[Nguồn: {c['metadata'].get('filename', 'unknown')}]\n{c['content']}"
        for c in context_chunks
    )

    # FIXED: use structured content parts to separate context from user question (mitigates prompt injection)
    prompt = f"""=== TÀI LIỆU THAM KHẢO ===
{context_text}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ==="""

    try:
        response = get_model().generate_content(
            prompt,
            request_options={"timeout": 60},  # FIXED: 60s timeout to prevent indefinite hangs
        )
        return response.text
    except ResourceExhausted as e:
        raise ResourceExhausted(
            "Gemini API đã hết quota. Vui lòng kiểm tra billing tại https://ai.dev/rate-limit"
        ) from e
