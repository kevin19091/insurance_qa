"""Tests for serving/factory.py's load_index() — read-only index connection."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.config import PipelineConfig
from src.pipeline.ingestion.pipeline import run_ingestion
from src.pipeline.serving.factory import load_index
from tests.conftest import requires_qdrant

_TEST_QDRANT_COLLECTION = "test_load_index"


@pytest.mark.slow
class TestLoadIndexChroma:
    def _config(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(storage={"chroma_path": str(tmp_path / "chroma")})  # type: ignore[arg-type]

    def test_raises_when_never_ingested(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        with pytest.raises(RuntimeError, match="does not exist"):
            load_index(config)

    def test_loads_queryable_index_after_ingestion(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        run_ingestion(config)
        index = load_index(config)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("insurance coverage")
        assert len(nodes) > 0


@requires_qdrant
@pytest.mark.slow
class TestLoadIndexQdrant:
    @pytest.fixture(autouse=True)
    def _cleanup(self) -> Generator[None, None, None]:
        yield
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists(_TEST_QDRANT_COLLECTION):
            client.delete_collection(_TEST_QDRANT_COLLECTION)

    def _config(self) -> PipelineConfig:
        return PipelineConfig(
            storage={"backend": "qdrant", "collection_name": _TEST_QDRANT_COLLECTION}  # type: ignore[arg-type]
        )

    def test_raises_when_never_ingested(self) -> None:
        with pytest.raises(RuntimeError, match="does not exist"):
            load_index(self._config())

    def test_loads_queryable_index_after_ingestion(self) -> None:
        config = self._config()
        run_ingestion(config)
        index = load_index(config)
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("insurance coverage")
        assert len(nodes) > 0
