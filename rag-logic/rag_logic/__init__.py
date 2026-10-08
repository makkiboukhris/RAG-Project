"""
RAG logic: chunk source code, embed the chunks, search them in plain English.

Everything a backend needs is importable from here:

    from rag_logic import build_index, search, get_index_status, preload_model, RagError

    preload_model()                      # once, at startup (optional)

    stats = build_index("path/to/code")  # -> IndexStats
    update_file("path/to/code", "src/App.tsx")  # file edited/created -> FileUpdate
    delete_file("path/to/code", "src/Old.tsx")  # file deleted/renamed -> FileUpdate
    results = search("how is the button styled", top_k=5)  # -> list[SearchResult]
    status = get_index_status()          # -> IndexStatus

Return values are dataclasses (use dataclasses.asdict(), or return them
directly from a FastAPI endpoint). Failures raise a RagError subclass - see
errors.py for which HTTP status each one maps to.

Lower-level pieces, if an endpoint needs them:

    from rag_logic.chunker import chunk_source, chunk_file, chunk_folder
    from rag_logic.embedder import embed_documents, embed_query
"""

from .config import settings
from .embedder import download_model, is_model_installed, is_model_loaded, preload_model
from .errors import (
    BuildInProgressError,
    FolderNotFoundError,
    IndexMismatchError,
    IndexNotFoundError,
    InvalidRequestError,
    ModelNotInstalledError,
    NoChunksFoundError,
    RagError,
)
from .service import (
    FileUpdate,
    IndexStats,
    IndexStatus,
    SearchResult,
    build_index,
    delete_file,
    get_index_status,
    search,
    update_file,
)

__all__ = [
    "build_index",
    "search",
    "update_file",
    "delete_file",
    "get_index_status",
    "FileUpdate",
    "preload_model",
    "is_model_loaded",
    "is_model_installed",
    "download_model",
    "settings",
    "IndexStats",
    "IndexStatus",
    "SearchResult",
    "RagError",
    "BuildInProgressError",
    "FolderNotFoundError",
    "NoChunksFoundError",
    "IndexNotFoundError",
    "IndexMismatchError",
    "InvalidRequestError",
    "ModelNotInstalledError",
]
