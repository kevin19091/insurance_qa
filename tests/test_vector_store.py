"""Tests for vector store backend dispatch (Chroma / Qdrant)."""

import uuid
from typing import Any

import numpy as np
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

from src.config import PipelineConfig
from src.pipeline.common.vector_store import (
    build_vector_store,
    collection_exists,
    delete_collection,
)
from tests.conftest import requires_qdrant


def _node_with_embedding(text: str = "hello") -> TextNode:
    node = TextNode(text=text, id_=str(uuid.uuid4()))
    node.embedding = [0.1] * 1024
    return node


def _fake_embedding() -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array([[0.1] * 8], dtype=np.float32)


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


class TestChromaCollectionExists:
    def _config(self, tmp_path: object) -> PipelineConfig:
        return PipelineConfig(
            storage={"chroma_path": str(tmp_path), "collection_name": "test_exists_col"}  # type: ignore[arg-type]
        )

    def test_false_when_never_written(self, tmp_path: object) -> None:
        assert collection_exists(self._config(tmp_path)) is False

    def test_false_when_created_but_empty(self, tmp_path: object) -> None:
        """build_vector_store() uses get_or_create_collection() — merely
        constructing it must not make collection_exists report a false
        positive for a collection nobody has actually written to."""
        config = self._config(tmp_path)
        build_vector_store(config)
        assert collection_exists(config) is False

    def test_true_after_write(self, tmp_path: object) -> None:
        config = self._config(tmp_path)
        vector_store = build_vector_store(config)
        assert isinstance(vector_store, ChromaVectorStore)
        vector_store._collection.add(ids=["1"], embeddings=_fake_embedding(), documents=["hello"])
        assert collection_exists(config) is True

    def test_false_after_delete(self, tmp_path: object) -> None:
        config = self._config(tmp_path)
        vector_store = build_vector_store(config)
        assert isinstance(vector_store, ChromaVectorStore)
        vector_store._collection.add(ids=["1"], embeddings=_fake_embedding(), documents=["hello"])
        delete_collection(config)
        assert collection_exists(config) is False


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


@requires_qdrant
class TestQdrantCollectionExists:
    _COLLECTION = "test_exists_col_qdrant"

    def _config(self) -> PipelineConfig:
        return PipelineConfig(storage={"backend": "qdrant", "collection_name": self._COLLECTION})  # type: ignore[arg-type]

    def teardown_method(self) -> None:
        delete_collection(self._config())

    def test_false_when_never_written(self) -> None:
        assert collection_exists(self._config()) is False

    def test_true_after_write(self) -> None:
        config = self._config()
        vector_store = build_vector_store(config)
        vector_store.add([_node_with_embedding()])
        assert collection_exists(config) is True

    def test_false_after_delete(self) -> None:
        config = self._config()
        vector_store = build_vector_store(config)
        vector_store.add([_node_with_embedding()])
        delete_collection(config)
        assert collection_exists(config) is False
