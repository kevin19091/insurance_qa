"""Tests for query rewriter implementations."""
import pytest

from src.config import PipelineConfig
from src.pipeline.factory import build_rewriter


class TestNullQueryRewriter:
    def test_returns_query_as_single_element_list(self) -> None:
        config = PipelineConfig()
        rewriter = build_rewriter(config)
        result = rewriter.rewrite("Is cardiac surgery covered?")
        assert result == ["Is cardiac surgery covered?"]

    def test_works_when_explicitly_disabled(self) -> None:
        config = PipelineConfig(query_rewrite={"enabled": False})  # type: ignore[arg-type]
        rewriter = build_rewriter(config)
        result = rewriter.rewrite("What is the premium?")
        assert result == ["What is the premium?"]


class TestFactoryDispatch:
    def test_disabled_returns_null_rewriter(self) -> None:
        from src.pipeline.rewriter import NullQueryRewriter

        config = PipelineConfig()
        rewriter = build_rewriter(config)
        assert isinstance(rewriter, NullQueryRewriter)

    def test_unknown_strategy_raises(self) -> None:
        config = PipelineConfig.model_construct()
        config.query_rewrite.enabled = True
        config.query_rewrite.strategy = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown query rewrite strategy"):
            build_rewriter(config)
