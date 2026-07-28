"""Retrieval strategy implementations and rewriting-aware retrieval."""

from typing import Any, cast

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Prefetch, Rrf, RrfQuery
from rank_bm25 import BM25Okapi

from src.observability import observe
from src.pipeline import QueryRewriter, Reranker
from src.pipeline import Retriever as RetrieverABC
from src.pipeline.common.sparse_embedder import embed_query


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def extract_nodes_from_index(index: VectorStoreIndex) -> tuple[TextNode, ...]:
    """Extract all TextNode objects from an index, handling both fresh and persistent loads."""
    nodes_dict = getattr(index.index_struct, "nodes_dict", None)
    if nodes_dict:
        return tuple(nodes_dict.values())

    col = cast(ChromaVectorStore, index.vector_store)._collection
    result = col.get()
    nodes: list[TextNode] = []
    for i, doc_id in enumerate(result["ids"]):
        text = result["documents"][i] or "" if result["documents"] else ""
        meta = result["metadatas"][i] if result["metadatas"] else {}
        nodes.append(TextNode(text=text, id_=doc_id, metadata=meta))
    return tuple(nodes)


class IndexRetriever(RetrieverABC):
    def __init__(self, index: VectorStoreIndex, top_k: int) -> None:
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)


class BM25Retriever(RetrieverABC):
    """Sparse keyword retrieval using BM25."""

    def __init__(self, nodes: tuple[TextNode, ...], top_k: int = 5) -> None:
        self._nodes = nodes
        self._top_k = top_k
        corpus = [_tokenize(n.text) for n in nodes]
        self._bm25 = BM25Okapi(corpus)

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        tokenized = _tokenize(query.query_str)
        scores = self._bm25.get_scores(tokenized)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[: self._top_k]
        if not any(s > 0 for _, s in top):
            return []
        max_score = max(s for _, s in top if s > 0)
        return [NodeWithScore(node=self._nodes[i], score=float(s) / max_score) for i, s in top]


class QdrantNativeRetriever(RetrieverABC):
    """Dense/sparse/hybrid retrieval via Qdrant's own Query API.

    Deliberately bypasses QdrantVectorStore.query()'s built-in HYBRID mode —
    that fuses dense+sparse client-side via an alpha-weighted blend. This
    uses Qdrant's server-side weighted RRF (RrfQuery, requires Qdrant
    server >=v1.17.0) so fusion happens on the server and dense_weight/
    sparse_weight carry the same meaning they have for the Chroma+BM25
    HybridRetriever above.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        embed_model: BaseEmbedding,
        mode: str,
        top_k: int,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self._vector_store = cast(QdrantVectorStore, index.vector_store)
        self._client = self._vector_store.client
        self._collection_name = self._vector_store.collection_name
        self._dense_vector_name = self._vector_store.dense_vector_name
        self._sparse_vector_name = self._vector_store.sparse_vector_name
        self._embed_model = embed_model
        self._mode = mode
        self._top_k = top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        if self._mode == "dense":
            points = self._query_dense(query.query_str)
        elif self._mode == "sparse":
            points = self._query_sparse(query.query_str)
        elif self._mode == "hybrid":
            points = self._query_hybrid(query.query_str)
        else:
            msg = f"Unknown retrieval mode: {self._mode}"
            raise ValueError(msg)
        return self._points_to_nodes(points)

    def _query_dense(self, query_text: str) -> list[Any]:
        vector = self._embed_model.get_query_embedding(query_text)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            using=self._dense_vector_name,
            limit=self._top_k,
            with_payload=True,
        )
        return cast(list[Any], response.points)

    def _query_sparse(self, query_text: str) -> list[Any]:
        sparse_vector = embed_query(query_text)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=sparse_vector,
            using=self._sparse_vector_name,
            limit=self._top_k,
            with_payload=True,
        )
        return cast(list[Any], response.points)

    def _query_hybrid(self, query_text: str) -> list[Any]:
        dense_vector = self._embed_model.get_query_embedding(query_text)
        sparse_vector = embed_query(query_text)
        response = self._client.query_points(
            collection_name=self._collection_name,
            prefetch=[
                Prefetch(query=dense_vector, using=self._dense_vector_name, limit=self._top_k),
                Prefetch(query=sparse_vector, using=self._sparse_vector_name, limit=self._top_k),
            ],
            query=RrfQuery(rrf=Rrf(weights=[self._dense_weight, self._sparse_weight])),
            limit=self._top_k,
            with_payload=True,
        )
        return cast(list[Any], response.points)

    def _points_to_nodes(self, points: list[Any]) -> list[NodeWithScore]:
        result = self._vector_store.parse_to_query_result(points)
        return [
            NodeWithScore(node=node, score=score)
            for node, score in zip(result.nodes or [], result.similarities or [], strict=True)
        ]


class SimilarityThresholdRetriever(RetrieverABC):
    """Drops nodes below a similarity-score floor.

    Only meaningful for dense-mode scores (a genuine 0-1 cosine similarity)
    — sparse and hybrid-fused scores aren't on a comparable scale, so this
    should only ever wrap a dense retriever, never sparse/hybrid.
    """

    def __init__(self, retriever: RetrieverABC, threshold: float) -> None:
        self._retriever = retriever
        self._threshold = threshold

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        nodes = self._retriever.retrieve(query)
        return [n for n in nodes if (n.score or 0) >= self._threshold]


class NullRetriever(RetrieverABC):
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return []


class HybridRetriever(RetrieverABC):
    """Fuse dense and sparse retrieval with weighted reciprocal rank fusion."""

    _RRF_K: float = 60.0

    def __init__(
        self,
        dense_retriever: RetrieverABC,
        sparse_retriever: RetrieverABC,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._top_k = top_k
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        dense_nodes = self._dense.retrieve(query)
        sparse_nodes = self._sparse.retrieve(query)

        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for rank, n in enumerate(dense_nodes):
            nid = n.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + self._dense_weight / (self._RRF_K + rank)
            node_map[nid] = n

        for rank, n in enumerate(sparse_nodes):
            nid = n.node.node_id
            rrf_scores[nid] = rrf_scores.get(nid, 0.0) + self._sparse_weight / (self._RRF_K + rank)
            node_map[nid] = n

        sorted_ids = sorted(rrf_scores.keys(), key=lambda nid: rrf_scores[nid], reverse=True)
        top_ids = sorted_ids[: self._top_k]
        max_score = max(rrf_scores[nid] for nid in top_ids) if top_ids else 1.0

        return [
            NodeWithScore(node=node_map[nid].node, score=rrf_scores[nid] / max_score)
            for nid in top_ids
        ]


def retrieve_with_rewriting(
    retriever: RetrieverABC,
    rewriter: QueryRewriter,
    query: QueryBundle,
) -> list[NodeWithScore]:
    """Rewrite query, retrieve for each variant, deduplicate by node ID."""
    rewritten = rewriter.rewrite(query.query_str)
    all_nodes: list[NodeWithScore] = []
    for rq in rewritten:
        all_nodes.extend(retriever.retrieve(QueryBundle(rq)))
    seen: set[str] = set()
    deduped: list[NodeWithScore] = []
    for n in all_nodes:
        nid = n.node.node_id
        if nid not in seen:
            seen.add(nid)
            deduped.append(n)
    return deduped


class RerankingRetriever(RetrieverABC):
    """Wraps a retriever with a reranker — fetches more nodes, re-ranks to top_n."""

    def __init__(
        self,
        retriever: RetrieverABC,
        reranker: Reranker,
        max_input_chunks: int,
        top_n: int,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._max_input_chunks = max_input_chunks
        self._top_n = top_n

    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        nodes = self._retriever.retrieve(query)
        return self._reranker.rerank(query.query_str, nodes, self._top_n)


__all__ = [
    "BM25Retriever",
    "IndexRetriever",
    "NullRetriever",
    "RerankingRetriever",
    "retrieve_with_rewriting",
]
