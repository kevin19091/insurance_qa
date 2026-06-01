"""Tests for retrieval and generation pipeline."""

import os

import pytest
from dotenv import load_dotenv
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config import PipelineConfig
from src.pipeline.factory import build_index, build_retriever

load_dotenv()

_OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))


class TestRetriever:
    def test_retrieves_relevant_nodes_for_query(self) -> None:
        config = PipelineConfig()
        index = build_index(config)
        retriever = build_retriever(index, config.retrieval.top_k)
        results = retriever.retrieve(QueryBundle("premium payment"))
        assert len(results) > 0
        assert results[0].score is not None


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestGenerator:
    def test_generates_answer_from_context(self) -> None:
        from src.pipeline.factory import build_generator

        config = PipelineConfig()
        generator = build_generator(config)
        context = [
            NodeWithScore(
                node=TextNode(text="Premium means the amount payable by the Master Policyholder."),
                score=0.95,
            ),
        ]
        answer = generator.generate("What is the premium?", context)
        assert len(answer) > 20
        assert "premium" in answer.lower()


@pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
class TestChatEndpoint:
    def test_chat_returns_answer_with_citations(self) -> None:
        from fastapi.testclient import TestClient

        from src.main import app

        with TestClient(app) as client:
            resp = client.get("/api/chat", params={"q": "What is the premium?"})
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["answer"]) > 20
            assert "premium" in data["answer"].lower()
            assert len(data.get("sources", [])) > 0
