"""Shared BM25-style sparse embedding for Qdrant native sparse/hybrid retrieval.

Used by both ingestion (embed each chunk's text once, at write time) and
serving (embed the user's query, at retrieve time) — the model is cached at
module scope so it's constructed once, not per call.
"""

from qdrant_client.models import SparseVector

from src.observability import observe

_SPARSE_MODEL_NAME = "Qdrant/bm25"
_model: object | None = None


def get_sparse_model() -> object:
    """Return the cached fastembed Bm25 sparse embedder, constructing it once."""
    global _model
    if _model is None:
        from fastembed.sparse.bm25 import Bm25

        _model = Bm25(model_name=_SPARSE_MODEL_NAME)
    return _model


@observe(as_type="embedding")
def embed_documents(texts: list[str]) -> list[SparseVector]:
    if not texts:
        return []
    model = get_sparse_model()
    return [
        SparseVector(indices=list(e.indices), values=list(e.values))
        for e in model.embed(texts)  # type: ignore[attr-defined]
    ]


@observe(as_type="embedding")
def embed_query(text: str) -> SparseVector:
    model = get_sparse_model()
    embeddings = list(model.query_embed(text))  # type: ignore[attr-defined]
    e = embeddings[0]
    return SparseVector(indices=list(e.indices), values=list(e.values))


__all__ = ["embed_documents", "embed_query", "get_sparse_model"]
