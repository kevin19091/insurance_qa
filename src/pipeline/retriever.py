"""Retrieval strategy implementations."""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.observability import observe
from src.pipeline import Retriever as RetrieverABC


class IndexRetriever(RetrieverABC):
    def __init__(self, index: VectorStoreIndex, top_k: int) -> None:
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)


class NullRetriever(RetrieverABC):
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return []


__all__ = ["IndexRetriever", "NullRetriever"]
