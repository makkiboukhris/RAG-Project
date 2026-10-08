"""
Install the embedding model on disk (run once).

    python download_model.py          # skip if already installed
    python download_model.py --force  # re-download

The model goes to rag_logic.settings.model_path (override with RAG_MODEL_DIR).
After this, rag_logic loads it from disk and never goes to the internet.
"""

import argparse
import logging

from rag_logic import download_model, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--force", action="store_true", help="download again even if installed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = download_model(force=args.force)
    print(f"{settings.model_name} is installed at {path}")


if __name__ == "__main__":
    main()
