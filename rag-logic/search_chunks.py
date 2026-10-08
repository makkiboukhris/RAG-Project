"""
Search the chunk index built by embed_chunks.py using a plain-English query.

Usage:
    python search_chunks.py "how do I read a CSV file" --top-k 5
"""

import argparse
import logging
import sys

from rag_logic import RagError, search, settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Plain-English search query")
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        results = search(args.query, top_k=args.top_k)
    except RagError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    for rank, result in enumerate(results, 1):
        print(f"\n{'='*20} #{rank}  score={result.score:.4f}  "
              f"{result.file} [chunk {result.chunk_index}] {'='*20}\n")
        print(result.text)
