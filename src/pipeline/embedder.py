"""Embedding model implementations."""

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.pipeline import Embedder as EmbedderABC


class BgeEmbedder(EmbedderABC):
    def __init__(self, model_name: str, dimension: int) -> None:
        self._model: BaseEmbedding = HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=8,
        )
        self._dimension = dimension

    def embed(self, nodes: list[TextNode]) -> list[TextNode]:
        texts = [n.text for n in nodes]
        embeddings = self._model.get_text_embedding_batch(texts)
        for node, vec in zip(nodes, embeddings, strict=True):
            node.embedding = vec
        return nodes


__all__ = ["BgeEmbedder"]
