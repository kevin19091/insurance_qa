"""Ingestion pipeline: parse -> chunk -> embed -> write to the configured vector store.

Runs once, idempotently (skips if the collection is already populated, unless
force_rebuild). Never imported by serving code (src/main.py, src/api/).
"""

import os
from typing import Any, cast

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding

from src.config import PipelineConfig
from src.observability import observe
from src.pipeline import Chunker, Parser
from src.pipeline.common.embedder import build_embedder
from src.pipeline.common.vector_store import build_vector_store
from src.pipeline.ingestion.chunker import (
    AgenticChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from src.pipeline.ingestion.parser import PyMuPDFParser


def build_parser(config: PipelineConfig) -> Parser:
    return PyMuPDFParser()


def build_chunker(config: PipelineConfig, embed_model: Any | None = None) -> Chunker:
    strategy = config.chunk.strategy
    kw: dict[str, Any] = dict(
        chunk_size=config.chunk.chunk_size, chunk_overlap=config.chunk.chunk_overlap
    )
    if strategy == "recursive":
        return RecursiveChunker(**kw)
    if strategy == "sentence":
        return SentenceChunker(**kw)
    if strategy == "semantic":
        return SemanticChunker(**kw, embed_model=embed_model)
    if strategy == "agentic":
        return AgenticChunker(**kw, embed_model=embed_model)
    msg = f"Unknown chunk strategy: {strategy}"
    raise ValueError(msg)


def _collection_exists(config: PipelineConfig) -> bool:
    if config.storage.backend == "qdrant":
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        return bool(client.collection_exists(config.storage.collection_name))

    import chromadb

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
    existing = [c.name for c in chroma_client.list_collections()]
    return config.storage.collection_name in existing


def _delete_collection(config: PipelineConfig) -> None:
    if config.storage.backend == "qdrant":
        from qdrant_client import QdrantClient

        client = QdrantClient(
            url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            api_key=os.environ.get("QDRANT_API_KEY"),
        )
        client.delete_collection(config.storage.collection_name)
        return

    import chromadb

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)
    chroma_client.delete_collection(config.storage.collection_name)


@observe(as_type="span")
def run_ingestion(config: PipelineConfig, force_rebuild: bool = False) -> int:
    """Parse -> chunk -> embed -> write. Returns node count written (0 if skipped)."""
    exists = _collection_exists(config)
    if exists and not force_rebuild:
        return 0
    if exists and force_rebuild:
        _delete_collection(config)

    embedder = build_embedder(config)
    raw_embed_model = cast(BaseEmbedding, embedder.raw_model)

    parser = build_parser(config)
    chunker = build_chunker(config, embed_model=raw_embed_model)

    docs = parser.parse(config.ingestion.source_pdf)
    nodes = chunker.chunk(docs)

    vector_store = build_vector_store(config)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=raw_embed_model)

    return len(nodes)


__all__ = ["build_chunker", "build_parser", "run_ingestion"]
