"""Tests for generator prompt building and usage tracking."""

import os

import pytest
from llama_index.core.llms import MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from src.pipeline.generator import _build_prompt

_GOOGLE_API_AVAILABLE = bool(os.environ.get("GOOGLE_API_KEY"))


class TestBuildPrompt:
    def test_returns_chat_messages(self) -> None:
        node = NodeWithScore(
            node=TextNode(text="Premium is the amount payable.", metadata={"page": 8}),
            score=0.95,
        )
        messages = _build_prompt("What is premium?", [node])
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER

    def test_includes_context_in_prompt(self) -> None:
        node = NodeWithScore(
            node=TextNode(text="Premium is the amount payable.", metadata={"page": 8}),
            score=0.95,
        )
        messages = _build_prompt("What is premium?", [node])
        user_content = messages[1].content
        assert user_content is not None
        assert "Premium is the amount" in user_content
        assert "What is premium?" in user_content
        assert "[Source: Page 8]" in user_content

    def test_no_context_uses_different_prompt(self) -> None:
        messages = _build_prompt("What is premium?", [])
        assert len(messages) == 2
        sys_content = messages[0].content
        assert sys_content is not None
        assert "based ONLY on the provided context" not in sys_content
        assert "based on your knowledge" in sys_content
        user_content = messages[1].content
        assert user_content is not None
        assert "Context:" not in user_content
        assert user_content == "Question: What is premium?"

    def test_multiple_contexts(self) -> None:
        nodes = [
            NodeWithScore(
                node=TextNode(text="First chunk.", metadata={"page": 1}),
                score=0.9,
            ),
            NodeWithScore(
                node=TextNode(text="Second chunk.", metadata={"page": 2}),
                score=0.8,
            ),
        ]
        messages = _build_prompt("test", nodes)
        user_content = messages[1].content
        assert user_content is not None
        assert "First chunk." in user_content
        assert "Second chunk." in user_content
        assert "[Source: Page 1]" in user_content
        assert "[Source: Page 2]" in user_content


class TestUsageTracking:
    def test_usage_starts_at_zero(self) -> None:
        from src.pipeline.generator import ClaudeGenerator, OpenAIGenerator

        gens = [
            OpenAIGenerator(model="gpt-4o-mini", temperature=0, max_tokens=1024),
            ClaudeGenerator(model="claude-3-haiku", temperature=0, max_tokens=1024),
        ]
        for gen in gens:
            assert gen.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}  # type: ignore[attr-defined]


class TestFactoryDispatch:
    def _check(self, model: str, expected_cls: type) -> None:
        from src.config import LLMConfig, PipelineConfig
        from src.pipeline.factory import build_generator

        config = PipelineConfig(llm=LLMConfig(model=model))  # type: ignore[arg-type]
        gen = build_generator(config)
        assert isinstance(gen, expected_cls), (
            f"Expected {expected_cls.__name__} for {model}, got {type(gen).__name__}"
        )

    def test_openai_models(self) -> None:
        from src.pipeline.generator import OpenAIGenerator

        self._check("gpt-4o-mini", OpenAIGenerator)
        self._check("gpt-4o", OpenAIGenerator)

    def test_claude_models(self) -> None:
        from src.pipeline.generator import ClaudeGenerator

        self._check("claude-3.5-sonnet", ClaudeGenerator)
        self._check("claude-3-haiku", ClaudeGenerator)

    @pytest.mark.skipif(not _GOOGLE_API_AVAILABLE, reason="GOOGLE_API_KEY not set")
    def test_gemini_models(self) -> None:
        from src.pipeline.generator import GeminiGenerator

        self._check("gemini-2.0-flash", GeminiGenerator)
        self._check("gemini-1.5-pro", GeminiGenerator)

    def test_unknown_model_raises(self) -> None:
        import pytest

        from src.config import PipelineConfig
        from src.pipeline.factory import build_generator

        config = PipelineConfig()
        config.llm.model = "unknown-model"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown LLM model"):
            build_generator(config)
