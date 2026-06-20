"""Tests for reranker implementations."""
import os

import pytest
from dotenv import load_dotenv

from llama_index.core.schema import NodeWithScore, TextNode
from src.config import PipelineConfig
from src.pipeline.reranker import NullReranker, build_reranker

load_dotenv()

_COHERE_AVAILABLE = bool(os.environ.get("COHERE_API_KEY"))


class TestNullReranker:
    def test_passes_through_top_n_nodes(self) -> None:
        nodes = [
            NodeWithScore(node=TextNode(text="A", id_="a"), score=0.9),
            NodeWithScore(node=TextNode(text="B", id_="b"), score=0.8),
            NodeWithScore(node=TextNode(text="C", id_="c"), score=0.7),
        ]
        reranker = NullReranker()
        result = reranker.rerank("test", nodes, top_n=2)
        assert len(result) == 2
        assert result[0].node.node_id == "a"


class TestFactoryDispatch:
    def test_disabled_returns_null_reranker(self) -> None:
        config = PipelineConfig()
        result = build_reranker(config)
        assert isinstance(result, NullReranker)

    def test_unknown_model_raises(self) -> None:
        config = PipelineConfig.model_construct()
        config.reranker.enabled = True
        config.reranker.model = "nonexistent"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown reranker model"):
            build_reranker(config)

    @pytest.mark.skipif(not _COHERE_AVAILABLE, reason="COHERE_API_KEY not set")
    def test_cohere_enabled_returns_cohere_reranker(self) -> None:
        config = PipelineConfig.model_construct()
        config.reranker.enabled = True
        config.reranker.model = "cohere"
        result = build_reranker(config)
        from src.pipeline.reranker import CohereReranker

        assert isinstance(result, CohereReranker)

    @pytest.mark.slow
    def test_bge_reranker_enabled(self) -> None:
        config = PipelineConfig.model_construct()
        config.reranker.enabled = True
        config.reranker.model = "bge-reranker"
        result = build_reranker(config)
        from src.pipeline.reranker import CrossEncoderReranker

        assert isinstance(result, CrossEncoderReranker)

    @pytest.mark.slow
    def test_cross_encoder_enabled(self) -> None:
        config = PipelineConfig.model_construct()
        config.reranker.enabled = True
        config.reranker.model = "cross-encoder"
        result = build_reranker(config)
        from src.pipeline.reranker import CrossEncoderReranker

        assert isinstance(result, CrossEncoderReranker)
