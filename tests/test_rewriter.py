"""Tests for query rewriter implementations."""
import os

import pytest
from dotenv import load_dotenv

from src.config import PipelineConfig
from src.pipeline.factory import build_generator, build_rewriter

load_dotenv()
_OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))


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
        from src.pipeline.generator import OpenAIGenerator

        config = PipelineConfig.model_construct()
        config.query_rewrite.enabled = True
        config.query_rewrite.strategy = "nonexistent"  # type: ignore[assignment]
        generator = OpenAIGenerator(model="gpt-4o-mini", temperature=0.0, max_tokens=128)
        with pytest.raises(ValueError, match="Unknown query rewrite strategy"):
            build_rewriter(config, generator=generator)


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestHydeRewriterIntegration:
    def test_returns_hypothetical_answer(self) -> None:
        config = PipelineConfig(query_rewrite={"enabled": True, "strategy": "hyde"})  # type: ignore[arg-type]
        generator = build_generator(config)
        rewriter = build_rewriter(config, generator=generator)
        result = rewriter.rewrite("What is the maximum coverage amount?")
        assert len(result) == 1
        assert len(result[0]) > 20, "HyDE should produce a substantial hypothetical answer"


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestStepBackRewriterIntegration:
    def test_returns_broader_question(self) -> None:
        config = PipelineConfig(query_rewrite={"enabled": True, "strategy": "step-back"})  # type: ignore[arg-type]
        generator = build_generator(config)
        rewriter = build_rewriter(config, generator=generator)
        result = rewriter.rewrite("Is cardiac surgery covered?")
        assert len(result) == 1
        assert len(result[0]) > 10, "Step-back should produce a broader question"


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestMultiQueryRewriterIntegration:
    def test_returns_multiple_variants(self) -> None:
        config = PipelineConfig(query_rewrite={"enabled": True, "strategy": "multi-query"})  # type: ignore[arg-type]
        generator = build_generator(config)
        rewriter = build_rewriter(config, generator=generator)
        result = rewriter.rewrite("What is the premium?")
        assert len(result) >= 2, "Multi-query should produce at least 2 variants"
        assert all(len(q) > 0 for q in result)
