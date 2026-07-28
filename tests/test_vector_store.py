"""Tests for vector store backend dispatch (Chroma / Qdrant)."""

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config import PipelineConfig
from src.pipeline.common.vector_store import build_vector_store
from tests.conftest import requires_qdrant


class TestChromaBackend:
    def test_returns_chroma_vector_store(self, tmp_path: object) -> None:
        config = PipelineConfig(
            storage={
                "backend": "chroma",
                "chroma_path": str(tmp_path),
                "collection_name": "test_col",
            }  # type: ignore[arg-type]
        )
        vector_store = build_vector_store(config)
        assert isinstance(vector_store, ChromaVectorStore)
        assert vector_store._collection.name == "test_col"


@requires_qdrant
class TestQdrantBackend:
    def test_returns_qdrant_vector_store(self) -> None:
        config = PipelineConfig(storage={"backend": "qdrant", "collection_name": "test_col"})  # type: ignore[arg-type]
        vector_store = build_vector_store(config)
        assert isinstance(vector_store, QdrantVectorStore)
        assert vector_store.collection_name == "test_col"

    def test_defaults_to_localhost(self, monkeypatch: object) -> None:
        monkeypatch.delenv("QDRANT_URL", raising=False)  # type: ignore[attr-defined]
        config = PipelineConfig(storage={"backend": "qdrant"})  # type: ignore[arg-type]
        vector_store = build_vector_store(config)
        assert vector_store.client._client.rest_uri == "http://localhost:6333"

    def test_uses_url_from_env(self, monkeypatch: object) -> None:
        monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")  # type: ignore[attr-defined]
        config = PipelineConfig(storage={"backend": "qdrant"})  # type: ignore[arg-type]
        vector_store = build_vector_store(config)
        assert vector_store.client._client.rest_uri == "http://127.0.0.1:6333"

    def test_hybrid_enabled_with_bm25_sparse_model(self) -> None:
        config = PipelineConfig(storage={"backend": "qdrant"})  # type: ignore[arg-type]
        vector_store = build_vector_store(config)
        assert isinstance(vector_store, QdrantVectorStore)
        assert vector_store.enable_hybrid is True
        assert vector_store.fastembed_sparse_model == "Qdrant/bm25"
