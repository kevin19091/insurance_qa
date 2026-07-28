"""DEPRECATED — kept only so existing callers (main.py, run.py, some tests)
keep working while they're migrated to run_ingestion()/load_index(). Do not
add new usages; new code should import from src.pipeline.common,
src.pipeline.ingestion, or src.pipeline.serving directly.
"""

from typing import cast

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import PipelineConfig
from src.pipeline.common.embedder import build_embedder
from src.pipeline.ingestion.pipeline import build_chunker, build_parser
from src.pipeline.serving.factory import build_generator, build_retriever, build_rewriter


def build_index(config: PipelineConfig, force_rebuild: bool = False) -> VectorStoreIndex:
    collection_name = "insurance_policy"

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)

    existing_collections = [c.name for c in chroma_client.list_collections()]
    collection_exists = collection_name in existing_collections

    if not collection_exists or force_rebuild:
        if force_rebuild and collection_exists:
            chroma_client.delete_collection(collection_name)

        embedder = build_embedder(config)
        raw_embed_model = cast(BaseEmbedding, embedder.raw_model)

        parser = build_parser(config)
        chunker = build_chunker(config, embed_model=raw_embed_model)

        docs = parser.parse(config.ingestion.source_pdf)
        nodes = chunker.chunk(docs)

        chroma_collection = chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=raw_embed_model,
        )

    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embedder = build_embedder(config)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=cast(BaseEmbedding, embedder.raw_model),
    )


__all__ = [
    "build_chunker",
    "build_embedder",
    "build_generator",
    "build_index",
    "build_parser",
    "build_retriever",
    "build_rewriter",
]
