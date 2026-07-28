"""Vector store backend dispatch (Chroma / Qdrant), shared by ingestion and serving."""

import os

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from src.config import PipelineConfig


def build_vector_store(config: PipelineConfig) -> BasePydanticVectorStore:
    if config.storage.backend == "qdrant":
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        return QdrantVectorStore(
            client=client,
            collection_name=config.storage.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )

    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
    collection = chroma_client.get_or_create_collection(config.storage.collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def collection_exists(config: PipelineConfig) -> bool:
    """Whether the configured collection already has content (count > 0) —
    used by both ingestion (skip vs. re-ingest) and serving (fail fast vs.
    load). Deliberately content-based, not name-based: build_vector_store()
    calls Chroma's get_or_create_collection(), which would make a name-only
    check return a false positive for a collection nobody has written to."""
    if config.storage.backend == "qdrant":
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        if not client.collection_exists(config.storage.collection_name):
            return False
        return bool(client.count(config.storage.collection_name, exact=True).count > 0)

    import chromadb

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
    existing = [c.name for c in chroma_client.list_collections()]
    if config.storage.collection_name not in existing:
        return False
    collection = chroma_client.get_collection(config.storage.collection_name)
    return bool(collection.count() > 0)


def delete_collection(config: PipelineConfig) -> None:
    if config.storage.backend == "qdrant":
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        if client.collection_exists(config.storage.collection_name):
            client.delete_collection(config.storage.collection_name)
        return

    import chromadb

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
    existing = [c.name for c in chroma_client.list_collections()]
    if config.storage.collection_name in existing:
        chroma_client.delete_collection(config.storage.collection_name)


__all__ = ["build_vector_store", "collection_exists", "delete_collection"]
