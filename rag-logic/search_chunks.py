"""
Search the chunk index built by embed_chunks.py using a plain-English query.

Usage:
    python search_chunks.py "how do I read a CSV file" --top-k 5
"""

import argparse
import json

import numpy as np

from embed_chunks import MODEL_NAME, INDEX_METADATA_PATH, INDEX_EMBEDDINGS_PATH, load_model


def search(query, top_k=5):
    with open(INDEX_METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load(INDEX_EMBEDDINGS_PATH)  # (N, dim), already encoded as documents

    model = load_model()

    # Same task ("nl2code") as embed_chunks.py, but the QUERY-side prompt.
    query_embedding = model.encode(
        [query],
        prompt_name="nl2code_query",
        convert_to_numpy=True,
    )[0]

    # Cosine similarity: normalize both sides, then dot product.
    doc_norms = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_norm = query_embedding / np.linalg.norm(query_embedding)
    scores = doc_norms @ q_norm

    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "file": chunk["file"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Plain-English search query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    for rank, result in enumerate(search(args.query, top_k=args.top_k), 1):
        print(f"\n{'='*20} #{rank}  score={result['score']:.4f}  "
              f"{result['file']} [chunk {result['chunk_index']}] {'='*20}\n")
        print(result["text"])
