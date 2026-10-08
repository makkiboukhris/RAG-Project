"""
The functions a backend endpoint should call.

    build_index(folder)        chunk + embed a folder and store the result
    update_file(folder, file)  re-embed one new/edited file
    delete_file(folder, file)  drop one deleted/renamed file from the index
    search(query, top_k)       find the chunks that best match a query
    get_index_status()         is there an index, how big, is the model loaded

Each one takes plain arguments, returns a dataclass (or a list of them) that
serializes straight to JSON, and raises a RagError subclass on failure.
They are blocking - build_index in particular can run for minutes - so call
them from a normal `def` endpoint or a background task, not directly inside
an `async def`.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from . import embedder, index_store
from .chunker import (
    SkippedFile,
    chunk_folder,
    chunk_one_file,
    is_ignored_path,
    relative_file_path,
)
from .config import settings
from .errors import (
    BuildInProgressError,
    FolderNotFoundError,
    IndexMismatchError,
    IndexNotFoundError,
    InvalidRequestError,
    NoChunksFoundError,
)

logger = logging.getLogger(__name__)

# Only one index build at a time.
_build_lock = threading.Lock()
# Single-file updates/deletes run one at a time (they wait for each other).
_update_lock = threading.Lock()


@dataclass
class IndexStats:
    """Result of build_index()."""
    folder: str
    chunk_count: int
    file_count: int
    embedding_dim: int
    skipped_files: list[SkippedFile] = field(default_factory=list)


@dataclass
class FileUpdate:
    """Result of update_file() / delete_file()."""
    file: str            # path relative to the indexed folder
    action: str          # "added" | "updated" | "removed" | "unchanged" | "ignored" | "skipped"
    chunks_removed: int
    chunks_added: int
    chunk_count: int     # total chunks in the index afterwards
    skipped_reason: str | None = None


@dataclass
class SearchResult:
    score: float      # cosine similarity, higher is better
    file: str         # path relative to the indexed folder
    chunk_index: int  # position of the chunk within that file
    text: str


@dataclass
class IndexStatus:
    """Result of get_index_status()."""
    exists: bool
    chunk_count: int
    file_count: int
    embedding_dim: int | None
    updated_at: float | None  # unix timestamp of the last build
    model_name: str
    model_loaded: bool
    build_in_progress: bool


def build_index(folder_path, batch_size=None, show_progress=False) -> IndexStats:
    """Chunk every file under `folder_path`, embed the chunks and save the
    index, replacing the previous one.

    Raises FolderNotFoundError, NoChunksFoundError, or BuildInProgressError
    (if another build is already running).
    """
    folder_path = os.fspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FolderNotFoundError(f"Folder not found: {folder_path}")

    if not _build_lock.acquire(blocking=False):
        raise BuildInProgressError("An index build is already in progress.")
    try:
        chunked = chunk_folder(folder_path)
        if not chunked.chunks:
            raise NoChunksFoundError(f"No chunks found under {folder_path}")
        logger.info("Extracted %d chunks from %s", len(chunked.chunks), folder_path)

        embeddings = embedder.embed_documents(
            [c.text for c in chunked.chunks],
            batch_size=batch_size,
            show_progress=show_progress,
        )
        index = index_store.save_index([c.to_dict() for c in chunked.chunks], embeddings)
        logger.info("Saved %d embeddings (dim=%d) to %s",
                    index.chunk_count, index.embedding_dim, settings.index_dir)

        return IndexStats(
            folder=folder_path,
            chunk_count=index.chunk_count,
            file_count=index.file_count,
            embedding_dim=index.embedding_dim,
            skipped_files=chunked.skipped,
        )
    finally:
        _build_lock.release()


def _resolve_file(folder_path, file_path) -> tuple[str, str]:
    """Validate the arguments; returns (folder, path relative to it)."""
    folder_path = os.fspath(folder_path)
    if not os.path.isdir(folder_path):
        raise FolderNotFoundError(f"Folder not found: {folder_path}")
    try:
        return folder_path, relative_file_path(folder_path, file_path)
    except ValueError as e:
        raise InvalidRequestError(str(e)) from None


def _check_no_build():
    if _build_lock.locked():
        raise BuildInProgressError(
            "An index build is in progress; it will pick up this change itself "
            "or the file can be updated again once it's done."
        )


def delete_file(folder_path, file_path) -> FileUpdate:
    """Remove a file's chunks from the index (use when the file was deleted
    or renamed). `file_path` is absolute or relative to `folder_path`; the
    file does not have to exist any more.

    Raises FolderNotFoundError, InvalidRequestError (file outside the folder),
    IndexNotFoundError or BuildInProgressError.
    """
    folder_path, rel_path = _resolve_file(folder_path, file_path)
    with _update_lock:
        _check_no_build()
        index, removed = index_store.remove_file(rel_path)
    return FileUpdate(file=rel_path, action="removed" if removed else "unchanged",
                      chunks_removed=removed, chunks_added=0, chunk_count=index.chunk_count)


def update_file(folder_path, file_path, batch_size=None) -> FileUpdate:
    """Re-chunk and re-embed one file and replace its chunks in the index (use
    when the file was created or edited).

    Behaves like build_index() for a single file: a missing file, or one that
    can't be chunked, ends up with no chunks in the index; hidden/ignored paths
    are left alone. Only this file is embedded, so it's fast.

    Raises FolderNotFoundError, InvalidRequestError (file outside the folder),
    IndexNotFoundError, IndexMismatchError or BuildInProgressError.
    """
    folder_path, rel_path = _resolve_file(folder_path, file_path)
    if is_ignored_path(rel_path):
        return FileUpdate(file=rel_path, action="ignored", chunks_removed=0,
                          chunks_added=0, chunk_count=index_store.load_index().chunk_count)

    skipped_reason = None
    try:
        chunks = chunk_one_file(folder_path, rel_path)
    except FileNotFoundError:
        return delete_file(folder_path, rel_path)
    except Exception as e:
        logger.warning("Skipping %s: %s", rel_path, e)
        chunks, skipped_reason = [], str(e)

    # Embed outside the index lock - it's the slow part.
    embeddings = (embedder.embed_documents([c.text for c in chunks],
                                           batch_size=batch_size, show_progress=False)
                  if chunks else np.empty((0, 0)))

    with _update_lock:
        _check_no_build()
        index, removed = index_store.replace_file(
            rel_path, [c.to_dict() for c in chunks], embeddings
        )
    if skipped_reason:
        action = "skipped"
    elif not chunks:
        action = "removed" if removed else "unchanged"
    else:
        action = "updated" if removed else "added"
    return FileUpdate(file=rel_path, action=action, chunks_removed=removed,
                      chunks_added=len(chunks), chunk_count=index.chunk_count,
                      skipped_reason=skipped_reason)


def search(query: str, top_k: int | None = None) -> list[SearchResult]:
    """Return the `top_k` chunks most similar to a plain-English query,
    best match first.

    Raises InvalidRequestError (empty query / top_k < 1), IndexNotFoundError
    (nothing built yet) or IndexMismatchError (index built with another model).
    """
    if not isinstance(query, str) or not query.strip():
        raise InvalidRequestError("Query must not be empty.")
    if top_k is None:
        top_k = settings.default_top_k
    if top_k < 1:
        raise InvalidRequestError("top_k must be at least 1.")

    index = index_store.load_index()  # fails fast, before the model is loaded

    query_embedding = embedder.embed_query(query.strip()).astype(np.float32)
    if query_embedding.shape[0] != index.embedding_dim:
        raise IndexMismatchError(
            f"The index has {index.embedding_dim}-dim vectors but the current model "
            f"({settings.model_name}) produces {query_embedding.shape[0]}-dim vectors. "
            "Rebuild the index."
        )

    # Cosine similarity: both sides unit length, then dot product.
    norm = np.linalg.norm(query_embedding)
    if norm:
        query_embedding = query_embedding / norm
    scores = index.normalized @ query_embedding

    top_indices = np.argsort(-scores)[:top_k]
    return [
        SearchResult(
            score=float(scores[i]),
            file=index.chunks[i]["file"],
            chunk_index=index.chunks[i]["chunk_index"],
            text=index.chunks[i]["text"],
        )
        for i in top_indices
    ]


def get_index_status() -> IndexStatus:
    """Cheap status check - never loads the model."""
    try:
        index = index_store.load_index()
    except (IndexNotFoundError, IndexMismatchError):
        index = None

    return IndexStatus(
        exists=index is not None,
        chunk_count=index.chunk_count if index else 0,
        file_count=index.file_count if index else 0,
        embedding_dim=index.embedding_dim if index else None,
        updated_at=index.updated_at if index else None,
        model_name=settings.model_name,
        model_loaded=embedder.is_model_loaded(),
        build_in_progress=_build_lock.locked(),
    )
