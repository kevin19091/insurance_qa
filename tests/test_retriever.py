"""Tests for retriever and rewriting-aware retrieval."""
import os

import pytest
from dotenv import load_dotenv
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config import PipelineConfig
from src.pipeline.factory import build_generator, build_index, build_retriever, build_rewriter
from src.pipeline.retriever import retrieve_with_rewriting

load_dotenv()
_OPENAI_AVAILABLE = bool(os.environ.get("OPENAI_API_KEY"))


class TestRetrieveWithRewriting:
    def test_dedup_removes_duplicate_nodes(self) -> None:
        node_a = TextNode(text="Content A", id_="a")
        node_b = TextNode(text="Content B", id_="b")
        node_c = TextNode(text="Content C", id_="c")
        retriever = _FakeRetriever(
            [
                [NodeWithScore(node=node_a, score=0.9), NodeWithScore(node=node_b, score=0.8)],
                [NodeWithScore(node=node_c, score=0.7), NodeWithScore(node=node_a, score=0.9)],
            ]
        )
        rewriter = _MultiQueryRewriter(["q1", "q2"])

        result = retrieve_with_rewriting(retriever, rewriter, QueryBundle("test"))
        assert len(result) == 3
        assert {n.node.node_id for n in result} == {"a", "b", "c"}

    def test_preserves_order_with_dedup(self) -> None:
        node_a = TextNode(text="Content A", id_="a")
        node_b = TextNode(text="Content B", id_="b")
        node_c = TextNode(text="Content C", id_="c")
        retriever = _FakeRetriever(
            [
                [NodeWithScore(node=node_a, score=0.9), NodeWithScore(node=node_b, score=0.8)],
                [NodeWithScore(node=node_c, score=0.7), NodeWithScore(node=node_a, score=0.9)],
            ]
        )
        rewriter = _MultiQueryRewriter(["q1", "q2"])

        result = retrieve_with_rewriting(retriever, rewriter, QueryBundle("test"))
        assert [n.node.node_id for n in result] == ["a", "b", "c"]

    def test_single_variant_no_changes(self) -> None:
        node = TextNode(text="Content", id_="x")
        retriever = _FakeRetriever([[NodeWithScore(node=node, score=0.9)]])
        from src.pipeline.rewriter import NullQueryRewriter

        result = retrieve_with_rewriting(retriever, NullQueryRewriter(), QueryBundle("test"))
        assert len(result) == 1


class _FakeRetriever:
    def __init__(self, results_per_call: list[list[NodeWithScore]]) -> None:
        self._results = results_per_call
        self.call_count = 0

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        idx = self.call_count
        self.call_count += 1
        return self._results[idx % len(self._results)]


class _MultiQueryRewriter:
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    def rewrite(self, query: str) -> list[str]:
        return self._queries


class TestBM25Retriever:
    def test_returns_top_k_nodes(self) -> None:
        from src.pipeline.retriever import BM25Retriever

        nodes = [
            TextNode(text="The premium is payable monthly.", id_="a"),
            TextNode(text="Coverage includes cardiac surgery.", id_="b"),
            TextNode(text="Exclusions apply for pre-existing conditions.", id_="c"),
            TextNode(text="The policy covers hospitalisation expenses.", id_="d"),
            TextNode(text="Claim must be filed within 30 days.", id_="e"),
        ]
        retriever = BM25Retriever(nodes=tuple(nodes), top_k=2)
        result = retriever.retrieve(QueryBundle("cardiac surgery coverage hospitalisation"))
        assert len(result) == 2
        returned_ids = {n.node.node_id for n in result}
        assert "b" in returned_ids
        assert "d" in returned_ids

    def test_returns_empty_for_no_match(self) -> None:
        from src.pipeline.retriever import BM25Retriever

        nodes = [
            TextNode(text="The premium is payable monthly.", id_="p"),
            TextNode(text="Coverage includes cardiac surgery.", id_="c"),
            TextNode(text="Exclusions apply for pre-existing conditions.", id_="e"),
            TextNode(text="The policy covers hospitalisation expenses.", id_="h"),
            TextNode(text="Claim must be filed within 30 days.", id_="cl"),
        ]
        retriever = BM25Retriever(nodes=tuple(nodes), top_k=5)
        result = retriever.retrieve(QueryBundle("zzzzzzzzzzxxxxxxyyyyyy"))
        assert len(result) == 0

    def test_returns_matching_when_top_k_larger(self) -> None:
        from src.pipeline.retriever import BM25Retriever

        nodes = [
            TextNode(text="Premium payment terms.", id_="a"),
            TextNode(text="Coverage details for cardiac procedures.", id_="b"),
            TextNode(text="Exclusions for pre-existing conditions.", id_="c"),
            TextNode(text="Hospitalisation expense coverage.", id_="d"),
            TextNode(text="Claim filing procedure.", id_="e"),
        ]
        retriever = BM25Retriever(nodes=tuple(nodes), top_k=100)
        result = retriever.retrieve(QueryBundle("coverage cardiac hospitalisation"))
        assert len(result) >= 2


@pytest.mark.slow
class TestRetrieveWithRewritingIntegration:
    @pytest.mark.skipif(not _OPENAI_AVAILABLE, reason="OPENAI_API_KEY not set")
    def test_rewriting_retrieves_different_nodes(self) -> None:
        config = PipelineConfig(query_rewrite={"enabled": True, "strategy": "hyde"})  # type: ignore[arg-type]
        generator = build_generator(config)
        index = build_index(config)
        retriever = build_retriever(index, config.retrieval.top_k)
        rewriter = build_rewriter(config, generator=generator)

        direct = retriever.retrieve(QueryBundle("What is the maximum coverage amount?"))
        rewritten = retrieve_with_rewriting(retriever, rewriter, QueryBundle("What is the maximum coverage amount?"))

        assert len(direct) > 0
        assert len(rewritten) > 0
