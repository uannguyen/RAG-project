#!/usr/bin/env python3
"""
ingest.py — CLI script để ingest documents vào Qdrant.

Cách dùng:
  python scripts/ingest.py --dir data/documents
  python scripts/ingest.py --file path/to/file.pdf
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.ingest_pipeline import ingest_directory, ingest_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents vào RAG system")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", help="Thư mục chứa documents")
    group.add_argument("--file", help="File đơn lẻ cần ingest")
    args = parser.parse_args()

    if args.dir:
        ingest_directory(args.dir)
    else:
        ingest_file(args.file)
