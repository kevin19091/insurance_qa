"""Tests for the shared BM25-style sparse embedder used by Qdrant native sparse/hybrid retrieval."""

import pytest
from qdrant_client.models import SparseVector

from src.pipeline.common.sparse_embedder import embed_documents, embed_query, get_sparse_model


@pytest.mark.slow
class TestSparseModelCaching:
    def test_get_sparse_model_returns_cached_instance(self) -> None:
        first = get_sparse_model()
        second = get_sparse_model()
        assert first is second


@pytest.mark.slow
class TestEmbedDocuments:
    def test_returns_one_sparse_vector_per_document(self) -> None:
        result = embed_documents(["cardiac surgery is covered", "the premium is payable monthly"])
        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, SparseVector)
            assert len(vec.indices) > 0
            assert len(vec.indices) == len(vec.values)

    def test_empty_input_returns_empty_list(self) -> None:
        assert embed_documents([]) == []


@pytest.mark.slow
class TestEmbedQuery:
    def test_returns_single_sparse_vector(self) -> None:
        vec = embed_query("is cardiac surgery covered")
        assert isinstance(vec, SparseVector)
        assert len(vec.indices) > 0
        assert len(vec.indices) == len(vec.values)

    def test_shares_vocabulary_with_matching_document(self) -> None:
        doc_vec = embed_documents(["cardiac surgery coverage details"])[0]
        query_vec = embed_query("cardiac surgery")
        assert set(query_vec.indices) & set(doc_vec.indices)
