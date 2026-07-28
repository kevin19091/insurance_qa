"""Tests for run_ingestion() — the shared parse->chunk->embed->write pipeline."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.config import PipelineConfig
from src.pipeline.ingestion.pipeline import run_ingestion
from tests.conftest import requires_qdrant

_TEST_QDRANT_COLLECTION = "test_run_ingestion"


@pytest.mark.slow
class TestRunIngestionChroma:
    def _config(self, tmp_path: Path) -> PipelineConfig:
        return PipelineConfig(storage={"chroma_path": str(tmp_path / "chroma")})  # type: ignore[arg-type]

    def test_first_call_writes_nodes_and_returns_count(self, tmp_path: Path) -> None:
        count = run_ingestion(self._config(tmp_path))
        assert count > 0

    def test_second_call_skips_and_returns_zero(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        run_ingestion(config)
        assert run_ingestion(config) == 0

    def test_force_rebuild_reingests(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        first = run_ingestion(config)
        second = run_ingestion(config, force_rebuild=True)
        assert second == first


@requires_qdrant
@pytest.mark.slow
class TestRunIngestionQdrant:
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

    def test_first_call_writes_nodes_and_returns_count(self) -> None:
        count = run_ingestion(self._config())
        assert count > 0

    def test_second_call_skips_and_returns_zero(self) -> None:
        config = self._config()
        run_ingestion(config)
        assert run_ingestion(config) == 0

    def test_force_rebuild_reingests(self) -> None:
        config = self._config()
        first = run_ingestion(config)
        second = run_ingestion(config, force_rebuild=True)
        assert second == first
