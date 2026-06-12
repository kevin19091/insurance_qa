"""Embedding model implementations."""

from llama_index.core.schema import TextNode

from src.observability import observe
from src.pipeline import Embedder as EmbedderABC


class BgeEmbedder(EmbedderABC):
    def __init__(self, model_name: str, dimension: int) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._model = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=8,
        )
        self._dimension = dimension

    @property
    def raw_model(self) -> object:
        return self._model

    @observe(as_type="embedding")
    def embed(self, nodes: list[TextNode]) -> list[TextNode]:
        texts = [n.text for n in nodes]
        embeddings = self._model.get_text_embedding_batch(texts)
        for node, vec in zip(nodes, embeddings, strict=True):
            node.embedding = vec
        return nodes


class OpenAIEmbedder(EmbedderABC):
    def __init__(self, model_name: str, dimension: int) -> None:
        from llama_index.embeddings.openai import OpenAIEmbedding

        self._model = OpenAIEmbedding(
            model=model_name,
            dimensions=dimension if dimension != 1024 else None,
        )
        self._dimension = dimension

    @property
    def raw_model(self) -> object:
        return self._model

    @observe(as_type="embedding")
    def embed(self, nodes: list[TextNode]) -> list[TextNode]:
        texts = [n.text for n in nodes]
        embeddings = self._model.get_text_embedding_batch(texts)
        for node, vec in zip(nodes, embeddings, strict=True):
            node.embedding = vec
        return nodes


class CohereEmbedder(EmbedderABC):
    def __init__(self, model_name: str, dimension: int) -> None:
        from llama_index.embeddings.cohere import CohereEmbedding

        self._model = CohereEmbedding(
            model_name=model_name,
            input_type="search_document",
        )
        self._dimension = dimension

    @property
    def raw_model(self) -> object:
        return self._model

    @observe(as_type="embedding")
    def embed(self, nodes: list[TextNode]) -> list[TextNode]:
        texts = [n.text for n in nodes]
        embeddings = self._model.get_text_embedding_batch(texts)
        for node, vec in zip(nodes, embeddings, strict=True):
            node.embedding = vec
        return nodes


class E5Embedder(EmbedderABC):
    def __init__(self, model_name: str, dimension: int) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        self._model = HuggingFaceEmbedding(
            model_name="intfloat/e5-large-v2",
            embed_batch_size=8,
        )
        self._dimension = dimension

    @property
    def raw_model(self) -> object:
        return self._model

    @observe(as_type="embedding")
    def embed(self, nodes: list[TextNode]) -> list[TextNode]:
        texts = [n.text for n in nodes]
        embeddings = self._model.get_text_embedding_batch(texts)
        for node, vec in zip(nodes, embeddings, strict=True):
            node.embedding = vec
        return nodes


__all__ = ["BgeEmbedder", "CohereEmbedder", "E5Embedder", "OpenAIEmbedder"]
