"""Query rewriting implementations (Null, HyDE, multi-query, step-back)."""

from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole

from src.pipeline import Generator, QueryRewriter


class NullQueryRewriter(QueryRewriter):
    """No-op rewriter — passes the query through unchanged."""

    def rewrite(self, query: str) -> list[str]:
        return [query]


def _call_llm(generator: Generator, system_prompt: str, user_message: str) -> str:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(role=MessageRole.USER, content=user_message),
    ]
    from llama_index.core.llms import LLM

    llm = generator.llm
    assert isinstance(llm, LLM)
    response = llm.chat(messages)
    return response.message.content or ""


class HyDEQueryRewriter(QueryRewriter):
    """Hypothetical Document Embedding — generate a hypothetical answer document and use it as the query."""

    _PROMPT = (
        "You are an insurance policy expert. "
        "Given a question, write a short paragraph that answers it as if it "
        "were extracted directly from the policy document. "
        "Include specific details about coverage amounts, conditions, and page references."
    )

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def rewrite(self, query: str) -> list[str]:
        return [_call_llm(self._generator, self._PROMPT, query)]


class StepBackRewriter(QueryRewriter):
    """Generate a broader step-back question to retrieve more context."""

    _PROMPT = (
        "You are an expert at analyzing questions. "
        "Given a specific insurance policy question, generate a broader, "
        "more general question that captures the underlying intent. "
        "This broader question will be used to search a policy document for relevant context."
        "Return only the broadened question, nothing else."
    )

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def rewrite(self, query: str) -> list[str]:
        return [_call_llm(self._generator, self._PROMPT, query)]


class MultiQueryRewriter(QueryRewriter):
    """Generate multiple variants of the query to improve retrieval coverage."""

    _PROMPT = (
        "You are an expert at rephrasing questions for better search results. "
        "Generate 3 different versions of the following insurance policy question. "
        "Each version should use different wording, synonyms, or phrasing "
        "to capture the same intent. "
        "Return one version per line, without numbering."
    )

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def rewrite(self, query: str) -> list[str]:
        raw = _call_llm(self._generator, self._PROMPT, query)
        variants = [line.strip() for line in raw.strip().split("\n") if line.strip()]
        return variants if variants else [query]


__all__ = [
    "HyDEQueryRewriter",
    "MultiQueryRewriter",
    "NullQueryRewriter",
    "QueryRewriter",
    "StepBackRewriter",
]
