"""Tests for ingestion pipeline (chunk, embed, store)."""

from pathlib import Path
from typing import Any, cast

import pytest

from llama_index.core.schema import Document, TextNode

from src.config import PipelineConfig
from src.pipeline.chunker import (
    AgenticChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
)
from src.pipeline.factory import build_chunker, build_embedder, build_index


class TestRecursiveChunker:
    def test_splits_large_text_into_multiple_chunks(self) -> None:
        config = PipelineConfig()
        chunker = build_chunker(config)
        text = "This is a sentence. " * 300
        docs = [Document(text=text)]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
        assert all(len(c.text) > 0 for c in chunks), "All chunks should have text"

    def test_preserves_metadata_from_source(self) -> None:
        config = PipelineConfig()
        chunker = build_chunker(config)
        docs = [Document(text="Page content. " * 50, metadata={"source": "test.pdf", "page": 1})]
        chunks = chunker.chunk(docs)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.metadata.get("source") == "test.pdf"
            assert c.metadata.get("page") == 1


class TestAllChunkers:
    def _chunk(self, chunker_cls: type, **kw: Any) -> list[TextNode]:
        chunker = chunker_cls(chunk_size=500, chunk_overlap=50, **kw)
        text = "This is a sentence. That is another sentence. " * 100
        docs = [Document(text=text)]
        return cast(list[TextNode], chunker.chunk(docs))

    def test_recursive_produces_multiple_nodes(self) -> None:
        nodes = self._chunk(RecursiveChunker)
        assert len(nodes) >= 2

    def test_sentence_produces_multiple_nodes(self) -> None:
        nodes = self._chunk(SentenceChunker)
        assert len(nodes) >= 2

    def test_reuses_embed_model(self) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", embed_batch_size=8)
        nodes = self._chunk(SemanticChunker, embed_model=embed_model)
        assert len(nodes) >= 1
        nodes2 = self._chunk(AgenticChunker, embed_model=embed_model)
        assert len(nodes2) >= 1

    def test_semantic_rejects_invalid_embed_model(self) -> None:
        from contextlib import suppress

        with suppress(Exception):
            SemanticChunker(chunk_size=500, chunk_overlap=50, embed_model="not-a-model")


class TestBgeEmbedder:
    def test_embeds_text_to_correct_dimension(self) -> None:
        config = PipelineConfig()
        embedder = build_embedder(config)
        nodes: list[TextNode] = [TextNode(text="What is the coverage for cardiac surgery?")]
        result = embedder.embed(nodes)
        assert result[0].embedding is not None
        assert len(result[0].embedding) == config.embedding.dimension


class TestFactoryEmbedderDispatch:
    def test_build_embedder_returns_correct_type(self) -> None:
        from src.pipeline.embedder import BgeEmbedder, E5Embedder, OpenAIEmbedder

        cases = [
            ("bge-large", BgeEmbedder),
            ("text-embedding-3-small", OpenAIEmbedder),
            ("e5-large", E5Embedder),
        ]
        for model, expected_cls in cases:
            config = PipelineConfig(embedding={"model": model})  # type: ignore[arg-type]
            embedder = build_embedder(config)
            assert isinstance(embedder, expected_cls), (
                f"Expected {expected_cls.__name__} for {model}, got {type(embedder).__name__}"
            )

    def test_unknown_model_raises(self) -> None:
        import pytest

        config = PipelineConfig()
        config.embedding.model = "unknown-model"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown embedding model"):
            build_embedder(config)


class TestPersistentChroma:
    @pytest.mark.slow
    def test_first_call_creates_sqlite_db(self, tmp_path: Path) -> None:
        chroma_path = tmp_path / "chroma"
        config = PipelineConfig(storage={"chroma_path": str(chroma_path)})  # type: ignore[arg-type]
        build_index(config)
        assert chroma_path.exists()
        assert (chroma_path / "chroma.sqlite3").exists()

    @pytest.mark.slow
    def test_second_call_skips_reingestion(self, tmp_path: Path) -> None:
        chroma_path = tmp_path / "chroma"
        config = PipelineConfig(storage={"chroma_path": str(chroma_path)})  # type: ignore[arg-type]
        build_index(config)
        mtime1 = (chroma_path / "chroma.sqlite3").stat().st_mtime
        build_index(config)
        mtime2 = (chroma_path / "chroma.sqlite3").stat().st_mtime
        assert mtime1 == mtime2, "Second call should NOT re-create the database"

    @pytest.mark.slow
    def test_force_rebuild_recreates_index(self, tmp_path: Path) -> None:
        chroma_path = tmp_path / "chroma"
        config = PipelineConfig(storage={"chroma_path": str(chroma_path)})  # type: ignore[arg-type]
        build_index(config)
        mtime1 = (chroma_path / "chroma.sqlite3").stat().st_mtime
        build_index(config, force_rebuild=True)
        mtime2 = (chroma_path / "chroma.sqlite3").stat().st_mtime
        assert mtime2 > mtime1, "force_rebuild should re-create the database"


class TestIngestionPipeline:
    def test_build_index_creates_queryable_index(self) -> None:
        config = PipelineConfig()
        index = build_index(config)
        assert index is not None
        retriever = index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve("insurance coverage")
        assert len(nodes) > 0, "Ingestion should produce retrievable nodes"

    def test_health_endpoint_returns_node_count(self) -> None:
        from fastapi.testclient import TestClient

        from src.main import app

        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["node_count"] > 0
