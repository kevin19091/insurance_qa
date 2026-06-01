"""Configuration models for the pipeline.

One Pydantic model maps 1:1 to config.yaml.
"""

from typing import Literal

from pydantic import BaseModel


class ChunkConfig(BaseModel):
    strategy: Literal["recursive", "semantic", "sentence", "agentic"] = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


class EmbeddingConfig(BaseModel):
    model: Literal["bge-large", "text-embedding-3-small", "cohere-embed-v3", "e5-large"] = (
        "bge-large"
    )
    dimension: int = 1024


class RetrievalConfig(BaseModel):
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = 5
    similarity_threshold: float | None = None
    # Sparse config (BM25)
    sparse_weight: float = 0.3
    dense_weight: float = 0.7


class RerankerConfig(BaseModel):
    enabled: bool = False
    model: Literal["cohere", "bge-reranker", "cross-encoder"] = "cohere"
    top_n: int = 5
    max_input_chunks: int = 20


class QueryRewriteConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["hyde", "multi-query", "step-back"] = "hyde"


class LLMConfig(BaseModel):
    model: Literal["gpt-4o-mini", "gpt-4o", "claude-3.5-sonnet"] = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024


class CacheConfig(BaseModel):
    enabled: bool = False
    ttl_seconds: int = 3600
    max_entries: int = 1000


class PipelineConfig(BaseModel):
    chunk: ChunkConfig = ChunkConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()
    llm: LLMConfig = LLMConfig()
    cache: CacheConfig = CacheConfig()
    seed: int = 42
    prompt_version: str = "v1"
