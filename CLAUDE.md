# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start server (port 8000, auto-reload)
python3 run.py

# Start Qdrant vector DB (required before running server)
docker compose up -d

# Ingest a single file via CLI
python3 scripts/ingest.py --file data/documents/file.pdf

# Ingest all files in a directory
python3 scripts/ingest.py --dir data/documents

# Delete Qdrant collection (required after changing embedding model/dimensions)
python3 -c "from retrieval.vector_store import VectorStore; VectorStore().delete_collection()"
# or
curl -X DELETE http://localhost:6333/collections/documents
```

## Architecture

The pipeline flows in one direction: **File → Loader → Chunker → Embedder → Qdrant** for ingestion, and **Query → Embedder → Qdrant search → Gemini → Answer** for retrieval.

- `api/server.py` — FastAPI entry point. The `/upload` endpoint saves to `data/documents/` then calls `ingest_file()` via `asyncio.to_thread()` (it's sync). The `/ask` endpoint is a plain sync handler (FastAPI runs it in a threadpool automatically).
- `pipeline/` — Orchestrators only. `ingest_file()` and `ingest_directory()` coordinate loader→chunker→embedder→store. `ask()` coordinates embed_query→search→generate_answer.
- `ingestion/embedder.py` — Uses `google-generativeai` SDK with model `models/gemini-embedding-001`. Output is **3072 dimensions**.
- `retrieval/vector_store.py` — Wraps Qdrant. Uses a **module-level singleton** `_get_client()` — never instantiate `QdrantClient` directly.
- `generation/llm.py` — Uses a **module-level singleton** `get_model()`. Model name is read from `GEMINI_MODEL` env var.

## Critical constraints

**Embedding dimensions are locked to the Qdrant collection.** `VECTOR_SIZE = 3072` in `vector_store.py` matches `models/gemini-embedding-001`. If the embedding model is changed, the existing collection must be deleted and all documents re-ingested.

**`ingest_file()` is synchronous.** It does network I/O (Gemini API + Qdrant). Always call it with `asyncio.to_thread()` from async handlers.

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. From https://aistudio.google.com |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model. Change here without touching code. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Collection name |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
