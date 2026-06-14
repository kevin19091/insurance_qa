"""Pipeline factory — builds RAG components from a PipelineConfig.

Each method returns an instance of the corresponding interface.
Swapping strategies = changing the config → factory returns a different implementation.
"""

from typing import Any

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import PipelineConfig
from src.pipeline import Chunker, Embedder, Generator, Parser, QueryRewriter, Retriever
from src.pipeline.chunker import AgenticChunker, RecursiveChunker, SemanticChunker, SentenceChunker
from src.pipeline.embedder import BgeEmbedder, CohereEmbedder, E5Embedder, OpenAIEmbedder
from src.pipeline.generator import ClaudeGenerator, GeminiGenerator, OpenAIGenerator
from src.pipeline.parser import PyMuPDFParser
from src.pipeline.retriever import IndexRetriever, NullRetriever
from src.pipeline.rewriter import NullQueryRewriter

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
    model = config.embedding.model
    dimension = config.embedding.dimension

    if model == "bge-large":
        return BgeEmbedder(model_name=_BGE_MODEL, dimension=dimension)
    if model == "text-embedding-3-small":
        return OpenAIEmbedder(model_name=model, dimension=dimension)
    if model == "cohere-embed-v3":
        return CohereEmbedder(model_name=model, dimension=dimension)
    if model == "e5-large":
        return E5Embedder(model_name=model, dimension=dimension)

    msg = f"Unknown embedding model: {model}"
    raise ValueError(msg)


def build_index(config: PipelineConfig, force_rebuild: bool = False) -> VectorStoreIndex:
    collection_name = "insurance_policy"

    chroma_client = chromadb.PersistentClient(path=config.storage.chroma_path)

    existing_collections = [c.name for c in chroma_client.list_collections()]
    collection_exists = collection_name in existing_collections

    if not collection_exists or force_rebuild:
        if force_rebuild and collection_exists:
            chroma_client.delete_collection(collection_name)

        embedder = build_embedder(config)
        raw_embed_model = embedder.raw_model

        parser = build_parser(config)
        chunker = build_chunker(config, embed_model=raw_embed_model)

        docs = parser.parse("data/max-life-group-credit-life-secure-policy-document-v1.pdf")
        nodes = chunker.chunk(docs)

        chroma_collection = chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=raw_embed_model,  # type: ignore[arg-type]
        )

    chroma_collection = chroma_client.get_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    embedder = build_embedder(config)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embedder.raw_model,  # type: ignore[arg-type]
    )


def build_generator(config: PipelineConfig) -> Generator:
    model: str = config.llm.model
    temperature: float = config.llm.temperature
    max_tokens: int = config.llm.max_tokens
    if model.startswith("gpt"):
        return OpenAIGenerator(model=model, temperature=temperature, max_tokens=max_tokens)
    if model.startswith("claude"):
        return ClaudeGenerator(model=model, temperature=temperature, max_tokens=max_tokens)
    if model.startswith("gemini"):
        return GeminiGenerator(model=model, temperature=temperature, max_tokens=max_tokens)
    msg = f"Unknown LLM model: {model}"
    raise ValueError(msg)


def build_retriever(index: VectorStoreIndex, top_k: int) -> Retriever:
    if top_k == 0:
        return NullRetriever()
    return IndexRetriever(index=index, top_k=top_k)


def build_rewriter(config: PipelineConfig, generator: Generator | None = None) -> QueryRewriter:
    if not config.query_rewrite.enabled:
        return NullQueryRewriter()
    msg = f"Unknown query rewrite strategy: {config.query_rewrite.strategy}"
    raise ValueError(msg)
