"""Core interfaces for the RAG pipeline.

Each component has a base protocol/ABC so strategies are swappable via config.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from llama_index.core.schema import Document, NodeWithScore, QueryBundle


class Parser(ABC):
    """Parse a PDF file path into llama-index Documents."""

    @abstractmethod
    def parse(self, file_path: str) -> list[Document]: ...


class Chunker(ABC):
    """Split Documents into Nodes (chunks)."""

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]: ...


class Embedder(ABC):
    """Embed text into vectors. Wraps an embedding model."""

    @abstractmethod
    def embed(self, nodes: list[Document]) -> list[Document]: ...


class Retriever(ABC):
    """Retrieve relevant nodes for a query."""

    @abstractmethod
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]: ...


class Reranker(ABC):
    """Re-rank retrieved nodes."""

    @abstractmethod
    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]: ...


class Generator(ABC):
    """Generate an answer from retrieved context."""

    @abstractmethod
    def generate(self, query: str, context_nodes: list[NodeWithScore]) -> str: ...

    @abstractmethod
    async def stream(
        self, query: str, context_nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]: ...


class QueryRewriter(ABC):
    """Rewrite a user query to improve retrieval quality."""

    @abstractmethod
    def rewrite(self, query: str) -> str: ...
