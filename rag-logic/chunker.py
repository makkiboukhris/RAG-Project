"""
Print the chunks the chunker produces for a folder (for eyeballing/testing).

Usage:
    python chunker.py                # defaults to test-files
    python chunker.py path/to/folder
"""

import argparse
import logging

from rag_logic.chunker import chunk_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default="test-files")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = chunk_folder(args.folder)
    for i, chunk in enumerate(result.chunks, 1):
        print(f"\n{'='*20} CHUNK {i} ({chunk.file}) {'='*20}\n")
        print(chunk.text)
