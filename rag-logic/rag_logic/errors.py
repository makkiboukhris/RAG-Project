"""
Exceptions raised by the RAG logic.

The functions in this package never print-and-return or call sys.exit; they
raise one of these instead. That lets an endpoint turn each one into the
right HTTP response, e.g.:

    FolderNotFoundError  -> 404
    NoChunksFoundError   -> 422
    IndexNotFoundError   -> 409 (index has to be built first)
    IndexMismatchError   -> 409 (index was built with a different model)
    BuildInProgressError -> 409 (try again when the current build is done)
    InvalidRequestError  -> 400
    ModelNotInstalledError -> 503 (run download_model.py once)
"""


class RagError(Exception):
    """Base class - catch this to handle every error from this package."""


class FolderNotFoundError(RagError):
    """The folder to index does not exist or is not a folder."""


class NoChunksFoundError(RagError):
    """The folder was read but produced no chunks to embed."""


class IndexNotFoundError(RagError):
    """No index on disk yet - build_index() has to run first."""


class IndexMismatchError(RagError):
    """The index on disk is unreadable or doesn't match the current model."""


class BuildInProgressError(RagError):
    """build_index() was called while another build is still running."""


class ModelNotInstalledError(RagError):
    """The embedding model isn't on disk - download_model() has to run first."""


class InvalidRequestError(RagError, ValueError):
    """Bad input, e.g. an empty query or top_k < 1."""
