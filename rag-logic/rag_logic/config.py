"""
Central settings for the RAG logic.

Every value can be overridden with an environment variable, so the backend
can be configured without touching code:

    RAG_MODEL_NAME   embedding model (default jinaai/jina-code-embeddings-1.5b)
    RAG_MODEL_DIR    folder the model is installed in and loaded from
                     (default: <project>/models/<model name>)
    RAG_INDEX_DIR    folder holding chunk_index.json + chunk_embeddings.npy
                     (default: the project folder, next to the CLI scripts)
    RAG_DEVICE       "cuda" / "cpu" (default: cuda if available)
    RAG_BATCH_SIZE   embedding batch size (default 16)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    model_name: str = "jinaai/jina-code-embeddings-1.5b"
    index_dir: Path = PROJECT_ROOT
    model_dir: Path | None = None  # None = PROJECT_ROOT/models/<model name>
    device: str | None = None  # None = pick automatically
    batch_size: int = 16
    default_top_k: int = 5

    # jina-code-embeddings is trained with task-specific instruction prefixes.
    # Chunks are embedded as the "document" side and plain-English queries as
    # the "query" side of the SAME task ("nl2code") - don't mix tasks.
    document_prompt: str = "nl2code_document"
    query_prompt: str = "nl2code_query"

    @property
    def model_path(self) -> Path:
        """Where the model is installed on disk."""
        if self.model_dir is not None:
            return self.model_dir
        return PROJECT_ROOT / "models" / self.model_name.replace("/", "--")

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunk_index.json"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "chunk_embeddings.npy"


def _load_settings() -> Settings:
    defaults = Settings()
    return Settings(
        model_name=os.environ.get("RAG_MODEL_NAME", defaults.model_name),
        index_dir=Path(os.environ.get("RAG_INDEX_DIR", defaults.index_dir)),
        model_dir=Path(os.environ["RAG_MODEL_DIR"]) if os.environ.get("RAG_MODEL_DIR") else None,
        device=os.environ.get("RAG_DEVICE") or None,
        batch_size=int(os.environ.get("RAG_BATCH_SIZE", defaults.batch_size)),
    )


settings = _load_settings()
