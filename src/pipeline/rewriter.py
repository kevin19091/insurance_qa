"""Query rewriting implementations (Null, HyDE, multi-query, step-back)."""

from src.pipeline import Generator, QueryRewriter


class NullQueryRewriter(QueryRewriter):
    """No-op rewriter — passes the query through unchanged."""

    def rewrite(self, query: str) -> list[str]:
        return [query]


__all__ = ["NullQueryRewriter", "QueryRewriter"]
