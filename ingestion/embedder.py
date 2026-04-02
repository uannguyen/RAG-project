"""
embedder.py — Tạo vector embeddings dùng Google Embedding API (free, không tốn RAM).
"""

import os
import re
import time
import logging
from typing import List

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# FIXED: fail fast if API key is missing instead of silent None
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set. Check your .env file.")
genai.configure(api_key=_api_key)

EMBEDDING_MODEL = "models/gemini-embedding-001"
# Free tier: 100 requests/min → 1 call per 0.65s to stay safely under the limit
_RATE_LIMIT_SLEEP = 0.65


def _parse_retry_delay(exc: ResourceExhausted) -> float:
    """Extract the retry_delay seconds the API asked us to wait."""
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(exc))
    return float(match.group(1)) + 5 if match else 60.0


def embed_texts(texts: List[str], batch_size: int = 20, max_retries: int = 5) -> List[List[float]]:
    """Chuyển list text thành list vectors qua Google API."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                all_vectors.extend(result["embedding"])
                # FIXED: sleep after every API call (not conditionally) to respect 100 RPM free tier
                time.sleep(_RATE_LIMIT_SLEEP)
                break
            except ResourceExhausted as e:
                if attempt == max_retries - 1:
                    raise
                # FIXED: honor the retry_delay the API returns (~50s), not a 1-2s backoff
                wait = _parse_retry_delay(e)
                logger.warning("Rate limit hit (attempt %d/%d), waiting %.0fs as requested by API...", attempt + 1, max_retries, wait)
                time.sleep(wait)
            except (ServiceUnavailable, ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Transient API error (attempt %d/%d), retrying in %ds: %s", attempt + 1, max_retries, wait, e)
                time.sleep(wait)
    return all_vectors


def embed_query(query: str) -> List[float]:
    """Embed một câu query duy nhất."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    return result["embedding"]
