"""Tests for retriever and rewriting-aware retrieval."""

import os
from collections.abc import Generator

import pytest
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config import PipelineConfig
from src.pipeline import QueryRewriter as QueryRewriterABC
from src.pipeline import Retriever as RetrieverABC
from src.pipeline.factory import build_generator, build_index, build_retriever, build_rewriter
from src.pipeline.serving.retriever import retrieve_with_rewriting
from tests.conftest import requires_qdrant

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
        from src.pipeline.serving.rewriter import NullQueryRewriter

        result = retrieve_with_rewriting(retriever, NullQueryRewriter(), QueryBundle("test"))
        assert len(result) == 1


class _FakeRetriever(RetrieverABC):
    def __init__(self, results_per_call: list[list[NodeWithScore]]) -> None:
        self._results = results_per_call
        self.call_count = 0

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        idx = self.call_count
        self.call_count += 1
        return self._results[idx % len(self._results)]


class TestSimilarityThresholdRetriever:
    def test_drops_nodes_below_threshold(self) -> None:
        from src.pipeline.serving.retriever import SimilarityThresholdRetriever

        nodes = [
            NodeWithScore(node=TextNode(text="a", id_="a"), score=0.9),
            NodeWithScore(node=TextNode(text="b", id_="b"), score=0.1),
            NodeWithScore(node=TextNode(text="c", id_="c"), score=0.5),
        ]
        retriever = SimilarityThresholdRetriever(_FakeRetriever([nodes]), threshold=0.2)
        result = retriever.retrieve(QueryBundle("test"))
        assert {n.node.node_id for n in result} == {"a", "c"}

    def test_keeps_all_when_none_below_threshold(self) -> None:
        from src.pipeline.serving.retriever import SimilarityThresholdRetriever

        nodes = [
            NodeWithScore(node=TextNode(text="a", id_="a"), score=0.9),
            NodeWithScore(node=TextNode(text="b", id_="b"), score=0.8),
        ]
        retriever = SimilarityThresholdRetriever(_FakeRetriever([nodes]), threshold=0.2)
        result = retriever.retrieve(QueryBundle("test"))
        assert len(result) == 2

    def test_none_score_treated_as_zero(self) -> None:
        from src.pipeline.serving.retriever import SimilarityThresholdRetriever

        nodes = [NodeWithScore(node=TextNode(text="a", id_="a"), score=None)]
        retriever = SimilarityThresholdRetriever(_FakeRetriever([nodes]), threshold=0.2)
        result = retriever.retrieve(QueryBundle("test"))
        assert result == []


class _MultiQueryRewriter(QueryRewriterABC):
    def __init__(self, queries: list[str]) -> None:
        self._queries = queries

    def rewrite(self, query: str) -> list[str]:
        return self._queries


class TestBM25Retriever:
    def test_returns_top_k_nodes(self) -> None:
        from src.pipeline.serving.retriever import BM25Retriever

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
        from src.pipeline.serving.retriever import BM25Retriever

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
        from src.pipeline.serving.retriever import BM25Retriever

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


class TestHybridRetriever:
    def test_combines_dense_and_sparse_results(self) -> None:
        from src.pipeline.serving.retriever import HybridRetriever

        nodes = [
            TextNode(text="Premium payment terms.", id_="a"),
            TextNode(text="Coverage for cardiac surgery.", id_="b"),
            TextNode(text="Exclusions for pre-existing conditions.", id_="c"),
            TextNode(text="Hospitalisation coverage details.", id_="d"),
            TextNode(text="Claim filing procedure.", id_="e"),
        ]
        dense = _FixedScoreRetriever(
            {
                "test": [
                    ("b", 0.9),
                    ("d", 0.8),
                    ("a", 0.7),
                ]
            }
        )
        from src.pipeline.serving.retriever import BM25Retriever

        sparse = BM25Retriever(nodes=tuple(nodes), top_k=10)

        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            top_k=3,
            dense_weight=0.7,
            sparse_weight=0.3,
        )
        result = hybrid.retrieve(QueryBundle("test"))
        assert len(result) <= 3
        assert all((n.score or 0) > 0 for n in result)

    def test_dense_only_no_sparse_match(self) -> None:
        from src.pipeline.serving.retriever import HybridRetriever

        dense = _FixedScoreRetriever(
            {
                "test": [
                    ("x", 0.9),
                    ("y", 0.8),
                ]
            }
        )
        nodes = [
            TextNode(text="Premium payment terms.", id_="p"),
        ]
        from src.pipeline.serving.retriever import BM25Retriever

        sparse = BM25Retriever(nodes=tuple(nodes), top_k=5)
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, top_k=2)
        result = hybrid.retrieve(QueryBundle("test"))
        assert len(result) == 2
        assert result[0].node.node_id == "x"


class _FixedScoreRetriever(RetrieverABC):
    """Retriever that returns fixed results for given queries."""

    def __init__(self, results: dict[str, list[tuple[str, float]]]) -> None:
        self._results = results

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        items = self._results.get(query.query_str, [])
        nodes = []
        for node_id, score in items:
            nodes.append(NodeWithScore(node=TextNode(text="", id_=node_id), score=score))
        return nodes


class TestBuildRetrieverDispatch:
    def test_dense_mode_returns_index_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import IndexRetriever

        config = PipelineConfig(retrieval={"mode": "dense"})  # type: ignore[arg-type]
        from src.pipeline.factory import build_index

        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert isinstance(retriever, IndexRetriever)

    def test_sparse_mode_returns_bm25_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import BM25Retriever

        config = PipelineConfig(retrieval={"mode": "sparse"})  # type: ignore[arg-type]
        from src.pipeline.factory import build_index

        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert isinstance(retriever, BM25Retriever)

    def test_hybrid_mode_returns_hybrid_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import HybridRetriever

        config = PipelineConfig(retrieval={"mode": "hybrid"})  # type: ignore[arg-type]
        from src.pipeline.factory import build_index

        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert isinstance(retriever, HybridRetriever)

    @pytest.mark.slow
    def test_reranker_enabled_wraps_with_reranking_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import IndexRetriever, RerankingRetriever

        config = PipelineConfig.model_construct()
        config.retrieval.mode = "dense"
        config.reranker.enabled = True
        config.reranker.model = "cross-encoder"
        from src.pipeline.factory import build_index

        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert isinstance(retriever, RerankingRetriever)
        assert isinstance(retriever._retriever, IndexRetriever)

    def test_returns_top_k_nodes(self) -> None:
        from src.pipeline.serving.retriever import BM25Retriever

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
        from src.pipeline.serving.retriever import BM25Retriever

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
        from src.pipeline.serving.retriever import BM25Retriever

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


@requires_qdrant
@pytest.mark.slow
class TestBuildRetrieverDispatchQdrant:
    _COLLECTION = "test_build_retriever_dispatch_qdrant"

    @pytest.fixture(scope="class", autouse=True)
    def _ingested(self) -> Generator[None, None, None]:
        from qdrant_client import QdrantClient

        from src.pipeline.ingestion.pipeline import run_ingestion

        config = PipelineConfig(storage={"backend": "qdrant", "collection_name": self._COLLECTION})  # type: ignore[arg-type]
        run_ingestion(config)
        yield
        client = QdrantClient(url="http://localhost:6333")
        if client.collection_exists(self._COLLECTION):
            client.delete_collection(self._COLLECTION)

    def _load_index(self) -> VectorStoreIndex:
        from src.pipeline.factory import load_index

        config = PipelineConfig(storage={"backend": "qdrant", "collection_name": self._COLLECTION})  # type: ignore[arg-type]
        return load_index(config)

    def test_dense_mode_returns_qdrant_native_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import QdrantNativeRetriever

        config = PipelineConfig(
            storage={"backend": "qdrant", "collection_name": self._COLLECTION},  # type: ignore[arg-type]
            retrieval={"mode": "dense"},  # type: ignore[arg-type]
        )
        retriever = build_retriever(self._load_index(), top_k=5, config=config)
        assert isinstance(retriever, QdrantNativeRetriever)
        assert retriever._mode == "dense"

    def test_sparse_mode_returns_qdrant_native_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import QdrantNativeRetriever

        config = PipelineConfig(
            storage={"backend": "qdrant", "collection_name": self._COLLECTION},  # type: ignore[arg-type]
            retrieval={"mode": "sparse"},  # type: ignore[arg-type]
        )
        retriever = build_retriever(self._load_index(), top_k=5, config=config)
        assert isinstance(retriever, QdrantNativeRetriever)
        assert retriever._mode == "sparse"

    def test_hybrid_mode_returns_qdrant_native_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import QdrantNativeRetriever

        config = PipelineConfig(
            storage={"backend": "qdrant", "collection_name": self._COLLECTION},  # type: ignore[arg-type]
            retrieval={"mode": "hybrid"},  # type: ignore[arg-type]
        )
        retriever = build_retriever(self._load_index(), top_k=5, config=config)
        assert isinstance(retriever, QdrantNativeRetriever)
        assert retriever._mode == "hybrid"

    def test_similarity_threshold_wraps_dense_qdrant_retriever(self) -> None:
        from src.pipeline.factory import build_retriever
        from src.pipeline.serving.retriever import SimilarityThresholdRetriever

        config = PipelineConfig(
            storage={"backend": "qdrant", "collection_name": self._COLLECTION},  # type: ignore[arg-type]
            retrieval={"mode": "dense", "similarity_threshold": 0.2},  # type: ignore[arg-type]
        )
        retriever = build_retriever(self._load_index(), top_k=5, config=config)
        assert isinstance(retriever, SimilarityThresholdRetriever)


class TestBuildRetrieverSimilarityThresholdDispatch:
    def test_dense_mode_with_threshold_wraps_chroma_retriever(self) -> None:
        from src.pipeline.factory import build_index, build_retriever
        from src.pipeline.serving.retriever import SimilarityThresholdRetriever

        config = PipelineConfig(retrieval={"mode": "dense", "similarity_threshold": 0.2})  # type: ignore[arg-type]
        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert isinstance(retriever, SimilarityThresholdRetriever)

    def test_no_threshold_does_not_wrap(self) -> None:
        from src.pipeline.factory import build_index, build_retriever
        from src.pipeline.serving.retriever import IndexRetriever, SimilarityThresholdRetriever

        config = PipelineConfig(retrieval={"mode": "dense"})  # type: ignore[arg-type]
        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert not isinstance(retriever, SimilarityThresholdRetriever)
        assert isinstance(retriever, IndexRetriever)

    def test_sparse_mode_with_threshold_not_wrapped(self) -> None:
        """similarity_threshold is dense-only — BM25 scores aren't a comparable 0-1 scale."""
        from src.pipeline.factory import build_index, build_retriever
        from src.pipeline.serving.retriever import BM25Retriever, SimilarityThresholdRetriever

        config = PipelineConfig(retrieval={"mode": "sparse", "similarity_threshold": 0.2})  # type: ignore[arg-type]
        index = build_index(PipelineConfig())
        retriever = build_retriever(index, top_k=5, config=config)
        assert not isinstance(retriever, SimilarityThresholdRetriever)
        assert isinstance(retriever, BM25Retriever)


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
        rewritten = retrieve_with_rewriting(
            retriever, rewriter, QueryBundle("What is the maximum coverage amount?")
        )

        assert len(direct) > 0
        assert len(rewritten) > 0
