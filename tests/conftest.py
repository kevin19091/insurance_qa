"""Test configuration and shared fixtures."""

import os

import pytest

from src.config import PipelineConfig


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig()


def _check_qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient

        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        QdrantClient(url=url, timeout=1).get_collections()
        return True
    except Exception:
        return False


QDRANT_AVAILABLE = _check_qdrant_available()
requires_qdrant = pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not reachable")
