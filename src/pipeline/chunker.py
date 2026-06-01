"""Chunking strategy implementations."""

from typing import cast

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

from src.pipeline import Chunker as ChunkerABC


class RecursiveChunker(ChunkerABC):
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: list[Document]) -> list[TextNode]:
        nodes = self._splitter.get_nodes_from_documents(documents)
        return cast(list[TextNode], nodes)


__all__ = ["RecursiveChunker"]
