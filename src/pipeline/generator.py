"""LLM generator implementations."""

from collections.abc import AsyncGenerator
from typing import cast

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI

from src.pipeline import Generator as GeneratorABC


def _build_prompt(query: str, context_nodes: list[NodeWithScore]) -> list[ChatMessage]:
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


class OpenAIGenerator(GeneratorABC):
    def __init__(self, model: str, temperature: float, max_tokens: int) -> None:
        self._llm = OpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    def generate(self, query: str, context_nodes: list[NodeWithScore]) -> str:
        messages = _build_prompt(query, context_nodes)
        response = self._llm.chat(messages)
        return response.message.content or ""

    async def stream(  # type: ignore[override]
        self, query: str, context_nodes: list[NodeWithScore]
    ) -> AsyncGenerator[str, None]:
        messages = _build_prompt(query, context_nodes)
        response = await self._llm.astream_chat(messages)
        async for chunk in response:
            yield chunk.delta or ""


__all__ = ["OpenAIGenerator"]
