"""
Embedding model wrapper.

The model is installed on disk once (download_model()) and always loaded from
there - loading never goes to the internet.

The model is big (1.5B parameters), so it is loaded ONCE per process, the
first time it is needed, and reused for every later call. A backend can call
preload_model() at startup so the first request doesn't pay the load time.

torch / sentence-transformers are imported lazily inside get_model(), so
importing this package (e.g. for chunking only) stays fast.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from .config import settings
from .errors import ModelNotInstalledError

logger = logging.getLogger(__name__)

_model = None
_load_lock = threading.Lock()
# One encode at a time: concurrent requests would otherwise compete for the
# same GPU memory.
_encode_lock = threading.Lock()


def is_model_installed() -> bool:
    """True if the model files are on disk (no network needed to load it)."""
    return (settings.model_path / "modules.json").is_file()


def download_model(force: bool = False) -> Path:
    """Download the model from Hugging Face into settings.model_path (one time).

    This is the ONLY place that touches the network. After it has run,
    get_model() loads purely from disk.
    """
    if is_model_installed() and not force:
        return settings.model_path

    from huggingface_hub import snapshot_download

    settings.model_path.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s...", settings.model_name, settings.model_path)
    snapshot_download(repo_id=settings.model_name, local_dir=settings.model_path)
    logger.info("Model installed.")
    return settings.model_path


def get_model():
    """Return the shared SentenceTransformer, loading it on first use."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:  # another thread may have loaded it meanwhile
                import torch
                from sentence_transformers import SentenceTransformer

                if not is_model_installed():
                    raise ModelNotInstalledError(
                        f"Model not found at {settings.model_path}. "
                        "Install it once with: python download_model.py"
                    )

                device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32
                }
                logger.info("Loading %s from %s on %s...", settings.model_name, settings.model_path, device)
                _model = SentenceTransformer(
                    str(settings.model_path),
                    model_kwargs=model_kwargs,
                    processor_kwargs={"padding_side": "left"},
                    device=device,
                )
                logger.info("Model loaded.")
    return _model


def preload_model():
    """Load the model now (call once at backend startup)."""
    get_model()


def is_model_loaded() -> bool:
    return _model is not None

def unload_model():
    """Free the model's memory. It reloads automatically on the next use."""
    global _model
    with _load_lock, _encode_lock:   # wait for any running embed/search to finish
        if _model is None:
            return
        _model = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def embed_documents(texts, batch_size=None, show_progress=True) -> np.ndarray:
    """Embed code chunks (the "document" side). Returns an (N, dim) array."""
    model = get_model()
    with _encode_lock:
        return model.encode(
            list(texts),
            prompt_name=settings.document_prompt,
            batch_size=batch_size or settings.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )


def embed_query(query: str) -> np.ndarray:
    """Embed one plain-English query (the "query" side). Returns a (dim,) array."""
    model = get_model()
    with _encode_lock:
        return model.encode(
            [query],
            prompt_name=settings.query_prompt,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
