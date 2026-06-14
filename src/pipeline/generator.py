"""LLM generator implementations."""

from collections.abc import AsyncGenerator
from typing import Any, cast

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from src.observability import observe
from src.pipeline import Generator as GeneratorABC


def _build_prompt(query: str, context_nodes: list[NodeWithScore]) -> list[ChatMessage]:
    if not context_nodes:
        return [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an AI assistant for Max Life Insurance. "
                    "Answer the question based on your knowledge. "
                    "This response is AI-generated and does not constitute "
                    "legal or insurance advice."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Question: {query}",
            ),
        ]

    context_lines: list[str] = []
    for n in context_nodes:
        node = cast(TextNode, n.node)
        page = node.metadata.get("page", "?")
        context_lines.append(f"[Source: Page {page}]\n{node.text}")
    context_text = "\n\n".join(context_lines)
    return [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are an AI assistant for Max Life Insurance. "
                "Answer the question based ONLY on the provided context. "
                "Cite the source page for each claim using [Source: Page N]. "
                "If the context doesn't contain the answer, say so clearly. "
                "This response is AI-generated and does not constitute legal or insurance advice."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=f"Context:\n{context_text}\n\nQuestion: {query}",
        ),
    ]


def _track_llm_usage(self_obj: Any, response: object) -> None:
    usage = getattr(response, "additional_kwargs", {})
    self_obj._total_prompt_tokens += usage.get("prompt_tokens", 0)
    self_obj._total_completion_tokens += usage.get("completion_tokens", 0)


class OpenAIGenerator(GeneratorABC):
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        from llama_index.llms.openai import OpenAI

        self._llm = OpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @property
    def llm(self) -> object:
        return self._llm

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    @observe(as_type="generation")
    def generate(self, query: str, context_nodes: list[NodeWithScore]) -> str:
        messages = _build_prompt(query, context_nodes)
        response = self._llm.chat(messages)
        _track_llm_usage(self, response)
        return response.message.content or ""

    @observe(as_type="generation")
    async def stream(  # type: ignore[override]
        self, query: str, context_nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        messages = _build_prompt(query, context_nodes)
        response = await self._llm.astream_chat(messages)
        async for chunk in response:
            yield chunk.delta or ""


class ClaudeGenerator(GeneratorABC):
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        from llama_index.llms.anthropic import Anthropic

        self._llm = Anthropic(model=model, temperature=temperature, max_tokens=max_tokens)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @property
    def llm(self) -> object:
        return self._llm

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    @observe(as_type="generation")
    def generate(self, query: str, context_nodes: list[NodeWithScore]) -> str:
        messages = _build_prompt(query, context_nodes)
        response = self._llm.chat(messages)
        _track_llm_usage(self, response)
        return response.message.content or ""

    @observe(as_type="generation")
    async def stream(  # type: ignore[override]
        self, query: str, context_nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        messages = _build_prompt(query, context_nodes)
        response = await self._llm.astream_chat(messages)
        async for chunk in response:
            yield chunk.delta or ""


class GeminiGenerator(GeneratorABC):
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        from llama_index.llms.google_genai import GoogleGenAI

        self._llm = GoogleGenAI(model=model, temperature=temperature, max_tokens=max_tokens)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

    @property
    def llm(self) -> object:
        return self._llm

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
        }

    @observe(as_type="generation")
    def generate(self, query: str, context_nodes: list[NodeWithScore]) -> str:
        messages = _build_prompt(query, context_nodes)
        response = self._llm.chat(messages)
        _track_llm_usage(self, response)
        return response.message.content or ""

    @observe(as_type="generation")
    async def stream(  # type: ignore[override]
        self, query: str, context_nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        messages = _build_prompt(query, context_nodes)
        response = await self._llm.astream_chat(messages)
        async for chunk in response:
            yield chunk.delta or ""


__all__ = ["ClaudeGenerator", "GeminiGenerator", "OpenAIGenerator"]
