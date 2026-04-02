#!/usr/bin/env python3
"""
run.py — Khởi động FastAPI server.
"""

import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")  # FIXED: default to loopback, not 0.0.0.0
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"  # FIXED: opt-in reload via env var
    uvicorn.run("api.server:app", host=host, port=port, reload=reload)
