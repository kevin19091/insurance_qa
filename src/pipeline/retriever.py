"""Retrieval strategy implementations and rewriting-aware retrieval."""

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.observability import observe
from src.pipeline import QueryRewriter
from src.pipeline import Retriever as RetrieverABC


class IndexRetriever(RetrieverABC):
    def __init__(self, index: VectorStoreIndex, top_k: int) -> None:
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    @observe(as_type="retriever")
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return self._retriever.retrieve(query)


class NullRetriever(RetrieverABC):
    def retrieve(self, query: QueryBundle) -> list[NodeWithScore]:
        return []


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


__all__ = ["IndexRetriever", "NullRetriever", "retrieve_with_rewriting"]
