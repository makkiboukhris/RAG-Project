"""
On-disk chunk index: chunk_index.json (chunk text + metadata) and
chunk_embeddings.npy (one vector per chunk, same order).

The loaded index is kept in memory so a search doesn't re-read the files on
every request. It is reloaded automatically if the files change on disk
(e.g. after a rebuild from the command line).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass

import numpy as np

from .config import settings
from .errors import IndexMismatchError, IndexNotFoundError


@dataclass
class ChunkIndex:
    chunks: list[dict]              # [{"file", "chunk_index", "text"}, ...]
    embeddings: np.ndarray          # (N, dim), as produced by the model
    normalized: np.ndarray          # (N, dim), unit length, for cosine similarity
    updated_at: float               # mtime of the embeddings file (unix seconds)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def file_count(self) -> int:
        return len({c["file"] for c in self.chunks})


_cache: ChunkIndex | None = None
_cache_stamp = None
# Re-entrant so replace_file() can hold it across its load + save.
_lock = threading.RLock()


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero vector just scores 0 instead of NaN
    return embeddings / norms


def _disk_stamp():
    """Identifies the current files on disk; changes whenever they do."""
    meta = os.stat(settings.metadata_path)
    emb = os.stat(settings.embeddings_path)
    return (meta.st_mtime_ns, meta.st_size, emb.st_mtime_ns, emb.st_size)


def index_exists() -> bool:
    return settings.metadata_path.is_file() and settings.embeddings_path.is_file()


def save_index(chunks: list[dict], embeddings: np.ndarray) -> ChunkIndex:
    """Write the index to disk and make it the in-memory index.

    Each file is written to a temporary name and then swapped in, so a
    search running at the same time never reads a half-written file.
    """
    global _cache, _cache_stamp

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings - they must match"
        )

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    meta_tmp = settings.metadata_path.with_name(settings.metadata_path.name + ".tmp")
    emb_tmp = settings.embeddings_path.with_name(settings.embeddings_path.name + ".tmp")

    with _lock:
        with open(emb_tmp, "wb") as f:
            np.save(f, embeddings)
        with open(meta_tmp, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        os.replace(emb_tmp, settings.embeddings_path)
        os.replace(meta_tmp, settings.metadata_path)

        _cache = ChunkIndex(
            chunks=chunks,
            embeddings=embeddings,
            normalized=_normalize(embeddings),
            updated_at=os.stat(settings.embeddings_path).st_mtime,
        )
        _cache_stamp = _disk_stamp()
        return _cache


def replace_file(file: str, chunks: list[dict], embeddings: np.ndarray) -> tuple[ChunkIndex, int]:
    """Swap every chunk of `file` (path relative to the indexed folder) for the
    given new ones; pass no chunks to just remove the file. Chunks of other
    files are untouched.

    Returns (new index, number of old chunks removed). Raises
    IndexNotFoundError if there is no index yet, IndexMismatchError if the
    new embeddings have a different dimension than the index.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"{len(chunks)} chunks but {len(embeddings)} embeddings - they must match"
        )

    with _lock:  # nothing else may change the index between our load and save
        index = load_index()

        if len(chunks) and embeddings.shape[1] != index.embedding_dim:
            raise IndexMismatchError(
                f"New embeddings are {embeddings.shape[1]}-dim but the index is "
                f"{index.embedding_dim}-dim. Rebuild the index."
            )

        keep = np.array([c["file"] != file for c in index.chunks], dtype=bool)
        removed = int((~keep).sum())

        new_chunks = [c for c, k in zip(index.chunks, keep) if k] + list(chunks)
        new_embeddings = index.embeddings[keep]
        if len(chunks):
            new_embeddings = np.concatenate(
                [new_embeddings, embeddings.astype(new_embeddings.dtype, copy=False)]
            )
        return save_index(new_chunks, new_embeddings), removed


def remove_file(file: str) -> tuple[ChunkIndex, int]:
    """Delete every chunk of `file`. Returns (new index, chunks removed)."""
    return replace_file(file, [], np.empty((0, 0)))


def load_index() -> ChunkIndex:
    """Return the index, reading it from disk only if needed.

    Raises IndexNotFoundError if it hasn't been built yet.
    """
    global _cache, _cache_stamp

    with _lock:
        if not index_exists():
            _cache, _cache_stamp = None, None
            raise IndexNotFoundError(
                f"No index found in {settings.index_dir}. Build it first."
            )

        stamp = _disk_stamp()
        if _cache is not None and stamp == _cache_stamp:
            return _cache

        with open(settings.metadata_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        embeddings = np.load(settings.embeddings_path)

        if embeddings.ndim != 2 or len(chunks) != embeddings.shape[0]:
            raise IndexMismatchError(
                f"Index files don't match: {len(chunks)} chunks vs embeddings of "
                f"shape {embeddings.shape}. Rebuild the index."
            )

        _cache = ChunkIndex(
            chunks=chunks,
            embeddings=embeddings,
            normalized=_normalize(embeddings),
            updated_at=os.stat(settings.embeddings_path).st_mtime,
        )
        _cache_stamp = stamp
        return _cache
