"""Retrieval strategy implementations."""

from typing import cast

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.pipeline import Retriever as RetrieverABC


class IndexRetriever(RetrieverABC):
    def __init__(self, index: VectorStoreIndex, top_k: int) -> None:
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return cast(list[NodeWithScore], self._retriever.retrieve(query))


__all__ = ["IndexRetriever"]
