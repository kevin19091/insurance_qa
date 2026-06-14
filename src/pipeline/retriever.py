"""Retrieval strategy implementations and rewriting-aware retrieval."""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from rank_bm25 import BM25Okapi

from src.observability import observe
from src.pipeline import QueryRewriter
from src.pipeline import Retriever as RetrieverABC


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def extract_nodes_from_index(index: VectorStoreIndex) -> tuple[TextNode, ...]:
    """Extract all TextNode objects from an index, handling both fresh and persistent loads."""
    nodes_dict = getattr(index.index_struct, "nodes_dict", None)
    if nodes_dict:
        return tuple(nodes_dict.values())

    col = index.vector_store._collection
    result = col.get()
    nodes: list[TextNode] = []
    for i, doc_id in enumerate(result["ids"]):
        text = result["documents"][i] or ""
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
        return [
            NodeWithScore(node=self._nodes[i], score=float(s) / max_score)
            for i, s in top
        ]


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


__all__ = ["BM25Retriever", "IndexRetriever", "NullRetriever", "retrieve_with_rewriting"]
