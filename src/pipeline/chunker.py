"""Chunking strategy implementations."""

from typing import cast

from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)
from llama_index.core.schema import Document, TextNode

from src.observability import observe
from src.pipeline import Chunker as ChunkerABC


class RecursiveChunker(ChunkerABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @observe(as_type="span")
    def chunk(self, documents: list[Document]) -> list[TextNode]:
        nodes = self._splitter.get_nodes_from_documents(documents)
        return cast(list[TextNode], nodes)


class SentenceChunker(ChunkerABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @observe(as_type="span")
    def chunk(self, documents: list[Document]) -> list[TextNode]:
        nodes = self._splitter.get_nodes_from_documents(documents)
        return cast(list[TextNode], nodes)


class SemanticChunker(ChunkerABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
        self._splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=chunk_size,
            breakpoint_percentile_threshold=95,
        )

    @observe(as_type="span")
    def chunk(self, documents: list[Document]) -> list[TextNode]:
        nodes = self._splitter.get_nodes_from_documents(documents)
        return cast(list[TextNode], nodes)


class AgenticChunker(ChunkerABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        embed_model = resolve_embed_model("local:BAAI/bge-large-en-v1.5")
        self._splitter = SemanticDoubleMergingSplitterNodeParser(
            embed_model=embed_model,
            max_chunk_size=chunk_size,
        )

    @observe(as_type="span")
    def chunk(self, documents: list[Document]) -> list[TextNode]:
        nodes = self._splitter.get_nodes_from_documents(documents)
        return cast(list[TextNode], nodes)


__all__ = ["AgenticChunker", "RecursiveChunker", "SemanticChunker", "SentenceChunker"]
