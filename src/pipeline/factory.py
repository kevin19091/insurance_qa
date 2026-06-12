"""Pipeline factory — builds RAG components from a PipelineConfig.

Each method returns an instance of the corresponding interface.
Swapping strategies = changing the config → factory returns a different implementation.
"""

from typing import Any

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import PipelineConfig
from src.pipeline import Chunker, Embedder, Generator, Parser, Retriever
from src.pipeline.chunker import AgenticChunker, RecursiveChunker, SemanticChunker, SentenceChunker
from src.pipeline.embedder import BgeEmbedder
from src.pipeline.generator import OpenAIGenerator
from src.pipeline.parser import PyMuPDFParser
from src.pipeline.retriever import IndexRetriever, NullRetriever

# BGE-large-en-v1.5 (1024-dim, English)
_BGE_MODEL = "BAAI/bge-large-en-v1.5"


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


def build_embedder(config: PipelineConfig) -> Embedder:
    return BgeEmbedder(
        model_name=_BGE_MODEL,
        dimension=config.embedding.dimension,
    )


def build_index(config: PipelineConfig) -> VectorStoreIndex:
    parser = build_parser(config)

    embed_model = HuggingFaceEmbedding(
        model_name=_BGE_MODEL,
        embed_batch_size=8,
    )

    chunker = build_chunker(config, embed_model=embed_model)

    docs = parser.parse("data/max-life-group-credit-life-secure-policy-document-v1.pdf")
    nodes = chunker.chunk(docs)

    chroma_client = chromadb.EphemeralClient()
    for col in chroma_client.list_collections():
        if col.name == "insurance_policy":
            chroma_client.delete_collection("insurance_policy")
    chroma_collection = chroma_client.create_collection("insurance_policy")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_store,
        embed_model=embed_model,
    )


def build_generator(config: PipelineConfig) -> Generator:
    return OpenAIGenerator(
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )


def build_retriever(index: VectorStoreIndex, top_k: int) -> Retriever:
    if top_k == 0:
        return NullRetriever()
    return IndexRetriever(index=index, top_k=top_k)
