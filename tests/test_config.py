"""Tests for PipelineConfig and factory."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import (
    CacheConfig,
    ChunkConfig,
    EmbeddingConfig,
    LLMConfig,
    PipelineConfig,
    QueryRewriteConfig,
    RerankerConfig,
    RetrievalConfig,
)
from src.pipeline import Parser
from src.pipeline.factory import build_parser


class TestPipelineConfig:
    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.chunk == ChunkConfig()
        assert config.embedding == EmbeddingConfig()
        assert config.retrieval == RetrievalConfig()
        assert config.reranker == RerankerConfig()
        assert config.query_rewrite == QueryRewriteConfig()
        assert config.llm == LLMConfig()
        assert config.cache == CacheConfig()
        assert config.seed == 42
        assert config.prompt_version == "v1"

    def test_from_yaml(self) -> None:
        config_path = Path("benchmarks/M0/config.yaml")
        config = PipelineConfig.from_yaml(config_path)
        assert config.chunk == ChunkConfig()
        assert config.embedding == EmbeddingConfig()
        assert config.retrieval.top_k == 0  # M0 ablation disables retrieval
        assert config.retrieval.mode == "dense"
        assert config.reranker.enabled is False
        assert config.query_rewrite.enabled is False
        assert config.llm == LLMConfig()
        assert config.cache.enabled is False
        assert config.seed == 42
        assert config.prompt_version == "v1"

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(strategy="invalid-strategy")  # type: ignore[arg-type]


class TestFactory:
    def test_build_parser_returns_parser(self) -> None:
        config = PipelineConfig()
        parser = build_parser(config)
        assert isinstance(parser, Parser)
        result = parser.parse("data/max-life-group-credit-life-secure-policy-document-v1.pdf")
        assert len(result) > 0
        assert result[0].text is not None
        assert "Max Life" in result[0].text

    def test_build_chunker_returns_correct_type_by_strategy(self) -> None:
        from src.pipeline.chunker import (
            AgenticChunker,
            RecursiveChunker,
            SemanticChunker,
            SentenceChunker,
        )
        from src.pipeline.factory import build_chunker

        cases = [
            ("recursive", RecursiveChunker),
            ("sentence", SentenceChunker),
            ("semantic", SemanticChunker),
            ("agentic", AgenticChunker),
        ]
        for strategy, expected_cls in cases:
            cfg_dict = {"chunk": {"strategy": strategy}}
            cfg = PipelineConfig(**cfg_dict)  # type: ignore[arg-type]
            chunker = build_chunker(cfg)
            assert isinstance(chunker, expected_cls), (
                f"Expected {expected_cls.__name__} for strategy={strategy}, "
                f"got {type(chunker).__name__}"
            )

    def test_build_chunker_unknown_strategy_raises(self) -> None:
        from src.pipeline.factory import build_chunker

        config = PipelineConfig()
        config.chunk.strategy = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            build_chunker(config)
