"""
Chunk a folder of source files, embed the chunks with
jinaai/jina-code-embeddings-1.5b and store them on disk for later retrieval.

Install (CPU or GPU, GPU strongly recommended for 1.5B):
    pip install "sentence-transformers>=5.0.0" "torch>=2.7.1" einops

Usage:
    python embed_chunks.py /path/to/test-files
    # writes chunk_index.json + chunk_embeddings.npy next to this script

Then search with:
    python search_chunks.py "how do I read a CSV file"
"""

import argparse
import logging
import sys

from rag_logic import RagError, build_index, settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder of source files to chunk and embed")
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        stats = build_index(args.folder, batch_size=args.batch_size, show_progress=True)
    except RagError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"Indexed {stats.chunk_count} chunks from {stats.file_count} files "
          f"(dim={stats.embedding_dim}).")
    for skipped in stats.skipped_files:
        print(f"Skipped {skipped.file}: {skipped.reason}")
