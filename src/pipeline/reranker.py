"""Reranker implementations (Cohere, BGE-reranker, cross-encoder)."""

import os
from typing import Any, cast

from llama_index.core.schema import NodeWithScore, TextNode

from src.pipeline import Reranker


class NullReranker(Reranker):
    """No-op reranker — passes nodes through unchanged."""

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int) -> list[NodeWithScore]:
        return nodes[:top_n]


class CohereReranker(Reranker):
    """Rerank using Cohere's rerank API."""

    def __init__(self, model: str = "rerank-english-v3.0", top_n: int = 5) -> None:
        import cohere

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            msg = "COHERE_API_KEY is required for CohereReranker"
            raise ValueError(msg)
        self._client = cohere.Client(api_key)
        self._model = model
        self._top_n = top_n

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int | None = None) -> list[NodeWithScore]:
        k = top_n or self._top_n
        docs = [cast(TextNode, n.node).text for n in nodes]
        results = self._client.rerank(model=self._model, query=query, documents=docs, top_k=k)
        return [
            NodeWithScore(node=nodes[r.index].node, score=r.relevance_score)
            for r in results.results
        ]


class CrossEncoderReranker(Reranker):
    """Rerank using a local sentence-transformers cross-encoder model."""

    def __init__(self, model_name: str, top_n: int = 5) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._top_n = top_n

    def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int | None = None) -> list[NodeWithScore]:
        k = top_n or self._top_n
        texts = [cast(TextNode, n.node).text for n in nodes]
        pairs = [[query, t] for t in texts]
        scores = self._model.predict(pairs)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:k]
        max_score = max(s for _, s in top) if top else 1.0
        return [
            NodeWithScore(node=nodes[i].node, score=float(s) / max_score)
            for i, s in top
        ]


def build_reranker(config: Any) -> Reranker | None:
    """Build a Reranker from config. Returns None when disabled."""
    if not config.reranker.enabled:
        return NullReranker()

    model = config.reranker.model
    top_n = config.reranker.top_n

    if model == "cohere":
        return CohereReranker(top_n=top_n)
    if model == "bge-reranker":
        return CrossEncoderReranker(model_name="BAAI/bge-reranker-large", top_n=top_n)
    if model == "cross-encoder":
        return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n)

    msg = f"Unknown reranker model: {model}"
    raise ValueError(msg)


__all__ = ["CohereReranker", "CrossEncoderReranker", "NullReranker", "Reranker", "build_reranker"]
