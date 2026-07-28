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


__all__ = ["build_vector_store"]
