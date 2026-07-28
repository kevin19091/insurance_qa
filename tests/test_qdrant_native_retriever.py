"""Tests for QdrantNativeRetriever — dense/sparse/hybrid via Qdrant's native Query API."""

from collections.abc import Generator
from typing import cast

import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import QueryBundle
from qdrant_client import QdrantClient

from src.config import PipelineConfig
from src.pipeline.common.embedder import build_embedder
from src.pipeline.ingestion.pipeline import run_ingestion
from src.pipeline.serving.factory import load_index
from src.pipeline.serving.retriever import QdrantNativeRetriever
from tests.conftest import requires_qdrant

_TEST_COLLECTION = "test_qdrant_native_retriever"


@requires_qdrant
@pytest.mark.slow
class TestQdrantNativeRetriever:
    @pytest.fixture(scope="class", autouse=True)
    def _ingested(self) -> Generator[None, None, None]:
        config = PipelineConfig(
            storage={"backend": "qdrant", "collection_name": _TEST_COLLECTION}  # type: ignore[arg-type]
        )
        run_ingestion(config)
        yield
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists(_TEST_COLLECTION):
            client.delete_collection(_TEST_COLLECTION)

    def _config(self) -> PipelineConfig:
        return PipelineConfig(
            storage={"backend": "qdrant", "collection_name": _TEST_COLLECTION}  # type: ignore[arg-type]
        )

    def _retriever(self, mode: str, top_k: int = 5) -> QdrantNativeRetriever:
        config = self._config()
        index = load_index(config)
        embedder = build_embedder(config)
        return QdrantNativeRetriever(
            index=index,
            embed_model=cast(BaseEmbedding, embedder.raw_model),
            mode=mode,
            top_k=top_k,
        )

    def test_dense_mode_returns_relevant_nodes(self) -> None:
        retriever = self._retriever("dense")
        nodes = retriever.retrieve(QueryBundle("what does this policy cover"))
        assert len(nodes) > 0
        assert all(n.score is not None for n in nodes)

    def test_sparse_mode_returns_relevant_nodes(self) -> None:
        retriever = self._retriever("sparse")
        nodes = retriever.retrieve(QueryBundle("premium payment"))
        assert len(nodes) > 0
        assert all(n.score is not None for n in nodes)

    def test_hybrid_mode_returns_relevant_nodes(self) -> None:
        retriever = self._retriever("hybrid", top_k=5)
        nodes = retriever.retrieve(QueryBundle("credit life secure policy"))
        assert len(nodes) > 0
        assert all(n.score is not None for n in nodes)

    def test_respects_top_k(self) -> None:
        retriever = self._retriever("dense", top_k=2)
        nodes = retriever.retrieve(QueryBundle("policy"))
        assert len(nodes) <= 2

    def test_unknown_mode_raises(self) -> None:
        retriever = self._retriever("dense")
        retriever._mode = "nonexistent"
        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            retriever.retrieve(QueryBundle("test"))
