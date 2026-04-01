#!/usr/bin/env python3
"""
run.py — Khởi động FastAPI server.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
