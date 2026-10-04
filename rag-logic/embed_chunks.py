"""
Embed the chunks produced by chunker.py using jinaai/jina-code-embeddings-1.5b
and store them (vectors + metadata) on disk for later retrieval.

jina-code-embeddings is trained with task-specific instruction prefixes.
We use the "nl2code" task: chunks are embedded as the "document" side,
plain-English queries are embedded as the "query" side. Both sides must use
the SAME task, but different prompts - don't mix them.

Install (CPU or GPU, GPU strongly recommended for 1.5B):
    pip install "sentence-transformers>=5.0.0" "torch>=2.7.1" einops

Usage:
    python embed_chunks.py /path/to/test-files
    # writes chunk_index.json + chunk_embeddings.npy next to this script

Then search with:
    python search_chunks.py "how do I read a CSV file"
"""

import argparse
import json
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from chunker import extract_chunks_with_metadata

MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"

# Keep these two files in sync with search_chunks.py
INDEX_METADATA_PATH = "chunk_index.json"
INDEX_EMBEDDINGS_PATH = "chunk_embeddings.npy"


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(
        MODEL_NAME,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
        device=device,
    )
    return model


def embed_chunks(folder_path, batch_size=16):
    chunks = extract_chunks_with_metadata(folder_path)
    if not chunks:
        print(f"No chunks found under {folder_path}", file=sys.stderr)
        return

    print(f"Extracted {len(chunks)} chunks from {folder_path}")

    model = load_model()
    texts = [c["text"] for c in chunks]

    # "nl2code_document" is the sentence-transformers prompt name matching
    # the "Candidate code snippet:\n" instruction prefix for this task.
    embeddings = model.encode(
        texts,
        prompt_name="nl2code_document",
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(INDEX_EMBEDDINGS_PATH, embeddings)
    with open(INDEX_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Saved {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}) "
          f"to {INDEX_EMBEDDINGS_PATH}")
    print(f"Saved chunk metadata/text to {INDEX_METADATA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="Folder of source files to chunk and embed")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    embed_chunks(args.folder, batch_size=args.batch_size)
