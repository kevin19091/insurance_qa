"""Tests for pyproject.toml dependency declarations."""

import tomllib
from pathlib import Path


def _dependency_names() -> set[str]:
    raw = tomllib.loads(Path("pyproject.toml").read_text())
    deps = raw["project"]["dependencies"]
    return {dep.split(">")[0].split("<")[0].split("[")[0].strip() for dep in deps}


class TestQdrantDependencies:
    def test_qdrant_client_declared(self) -> None:
        assert "qdrant-client" in _dependency_names()

    def test_llama_index_qdrant_vector_store_declared(self) -> None:
        assert "llama-index-vector-stores-qdrant" in _dependency_names()

    def test_fastembed_declared(self) -> None:
        assert "fastembed" in _dependency_names()
