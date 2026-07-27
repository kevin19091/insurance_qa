"""Pre-download the local reranker models so they're served from the on-disk
Hugging Face cache instead of being fetched over the network at request time.

Run this once during image build / deploy, before the app starts serving traffic.

Usage:
    python -m src.download_models
"""

import sys

_MODELS = [
    "BAAI/bge-reranker-large",  # reranker.model: bge-reranker
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # reranker.model: cross-encoder
]


def main() -> None:
    from sentence_transformers import CrossEncoder

    for model_name in _MODELS:
        print(f"Downloading {model_name}...", file=sys.stderr)
        CrossEncoder(model_name)
        print(f"  cached: {model_name}", file=sys.stderr)


if __name__ == "__main__":
    main()
