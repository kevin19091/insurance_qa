"""Tests for retrieval and generation pipeline."""

import json
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

    def test_stream_returns_sse_events(self) -> None:
        from fastapi.testclient import TestClient

        from src.main import app

        with (
            TestClient(app) as client,
            client.stream("GET", "/api/chat/stream", params={"q": "What is the premium?"}) as resp,
        ):
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

            events: list[dict[str, str]] = []
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: "):
                    events.append({"event": line[7:], "data": ""})
                elif line.startswith("data: ") and events:
                    events[-1]["data"] = line[6:]

            assert len(events) >= 3
            assert events[0]["event"] == "sources"
            sources = json.loads(events[0]["data"])["sources"]
            assert len(sources) > 0

            tokens = [e for e in events if e["event"] == "token"]
            assert len(tokens) > 0
            full_text = "".join(json.loads(e["data"])["token"] for e in tokens)
            assert len(full_text) > 20

            assert events[-1]["event"] == "done"

    def test_abort_returns_ok(self) -> None:
        from fastapi.testclient import TestClient

        from src.main import app

        with TestClient(app) as client:
            resp = client.post("/api/chat/abort")
            assert resp.status_code == 200
            assert resp.json() == {"status": "aborted"}
