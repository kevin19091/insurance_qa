"""Serving-side factory — builds query-time RAG components from a PipelineConfig.

Never imports parser/chunker (ingestion-only) — only what src/main.py and
src/api/routes.py need at request time.
"""

from typing import cast

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding

from src.config import PipelineConfig
from src.pipeline import Generator, QueryRewriter, Retriever
from src.pipeline.common.embedder import build_embedder
from src.pipeline.common.vector_store import build_vector_store, collection_exists
from src.pipeline.serving.generator import ClaudeGenerator, GeminiGenerator, OpenAIGenerator
from src.pipeline.serving.retriever import (
    BM25Retriever,
    HybridRetriever,
    IndexRetriever,
    NullRetriever,
    RerankingRetriever,
    extract_nodes_from_index,
)
from src.pipeline.serving.rewriter import (
    HyDEQueryRewriter,
    MultiQueryRewriter,
    NullQueryRewriter,
    StepBackRewriter,
)


def load_index(config: PipelineConfig) -> VectorStoreIndex:
    """Connect to an already-populated vector store. Read-only — raises if
    the collection doesn't exist, rather than silently serving an empty index."""
    if not collection_exists(config):
        msg = (
            f"Collection {config.storage.collection_name!r} does not exist "
            f"(backend={config.storage.backend!r}). Run ingestion first."
        )
        raise RuntimeError(msg)

    vector_store = build_vector_store(config)
    embedder = build_embedder(config)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=cast(BaseEmbedding, embedder.raw_model),
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


def build_retriever(
    index: VectorStoreIndex,
    top_k: int,
    config: PipelineConfig | None = None,
) -> Retriever:
    if top_k == 0:
        return NullRetriever()

    if config is None:
        return IndexRetriever(index=index, top_k=top_k)

    # Build inner retriever — with larger top_k if reranker is enabled
    effective_top_k = config.reranker.max_input_chunks if config.reranker.enabled else top_k

    mode = config.retrieval.mode
    inner: Retriever
    if mode == "dense":
        inner = IndexRetriever(index=index, top_k=effective_top_k)
    elif mode == "sparse":
        nodes = extract_nodes_from_index(index)
        inner = BM25Retriever(nodes=nodes, top_k=effective_top_k)
    elif mode == "hybrid":
        nodes = extract_nodes_from_index(index)
        dense = IndexRetriever(index=index, top_k=effective_top_k)
        sparse = BM25Retriever(nodes=nodes, top_k=effective_top_k)
        inner = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            top_k=effective_top_k,
            dense_weight=config.retrieval.dense_weight,
            sparse_weight=config.retrieval.sparse_weight,
        )
    else:
        msg = f"Unknown retrieval mode: {mode}"
        raise ValueError(msg)

    if config.reranker.enabled:
        from src.pipeline.serving.reranker import build_reranker

        reranker = build_reranker(config)
        return RerankingRetriever(
            retriever=inner,
            reranker=reranker,
            max_input_chunks=config.reranker.max_input_chunks,
            top_n=config.reranker.top_n,
        )

    return inner


def build_rewriter(config: PipelineConfig, generator: Generator | None = None) -> QueryRewriter:
    if not config.query_rewrite.enabled:
        return NullQueryRewriter()

    if generator is None:
        msg = "Generator is required for non-null query rewriters"
        raise ValueError(msg)

    strategy = config.query_rewrite.strategy
    if strategy == "hyde":
        return HyDEQueryRewriter(generator)
    if strategy == "step-back":
        return StepBackRewriter(generator)
    if strategy == "multi-query":
        return MultiQueryRewriter(generator)

    msg = f"Unknown query rewrite strategy: {strategy}"
    raise ValueError(msg)


__all__ = ["build_generator", "build_retriever", "build_rewriter", "load_index"]
