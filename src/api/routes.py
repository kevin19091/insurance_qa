"""API routes for chat and health."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import suppress
from typing import Any, cast

from fastapi import APIRouter, Query, Request
from llama_index.core.schema import QueryBundle, TextNode
from sse_starlette.sse import EventSourceResponse

from src.pipeline.factory import build_generator, build_retriever, build_rewriter
from src.pipeline.retriever import retrieve_with_rewriting

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request) -> dict[str, str | int]:
    index = getattr(request.app.state, "index", None)
    if index is None:
        return {"status": "ok", "node_count": 0}
    col = index.vector_store._collection
    node_count = col.count()
    return {"status": "ok", "node_count": node_count}


@router.get("/chat")
async def chat(
    request: Request,
    q: str = Query(..., description="User question"),
) -> dict[str, str | list[dict[str, str | int | float | None]]]:
    index = getattr(request.app.state, "index", None)
    if index is None:
        return {"answer": "Index not available. Please ingest a document first.", "sources": []}

    retriever = build_retriever(index=index, top_k=5)
    generator = build_generator(request.app.state.config)
    rewriter = build_rewriter(request.app.state.config, generator=generator)
    config = request.app.state.config

    if config.query_rewrite.enabled:
        nodes = retrieve_with_rewriting(retriever, rewriter, QueryBundle(q))
    else:
        nodes = retriever.retrieve(QueryBundle(q))
    answer = generator.generate(q, nodes)

    sources = [
        {
            "page": n.node.metadata.get("page"),
            "text": cast(TextNode, n.node).text[:300],
            "score": round(n.score, 3) if n.score else None,
        }
        for n in nodes
    ]

    return {"answer": answer, "sources": sources}


def _build_sources(nodes: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "page": n.node.metadata.get("page"),
            "text": cast(TextNode, n.node).text[:300],
            "score": round(n.score, 3) if n.score else None,
        }
        for n in nodes
    ]


async def _stream_events(
    request: Request,
    generator_inst: Any,
    nodes: list[Any],
    sources: list[dict[str, Any]],
    q: str,
) -> AsyncGenerator[dict[str, str], None]:
    cancel_event = asyncio.Event()
    request.app.state.cancel_events.append(cancel_event)
    try:
        yield {"event": "sources", "data": json.dumps({"sources": sources})}
        async for token in generator_inst.stream(q, nodes):
            if cancel_event.is_set():
                yield {"event": "aborted", "data": json.dumps({"status": "aborted"})}
                return
            yield {"event": "token", "data": json.dumps({"token": token})}
        yield {"event": "done", "data": "[DONE]"}
    finally:
        with suppress(ValueError):
            request.app.state.cancel_events.remove(cancel_event)


async def _error_stream(msg: str) -> AsyncGenerator[dict[str, str], None]:
    yield {"event": "error", "data": json.dumps({"error": msg})}


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    q: str = Query(..., description="User question"),
) -> EventSourceResponse:
    index = getattr(request.app.state, "index", None)
    if index is None:
        return EventSourceResponse(_error_stream("No index available"))

    retriever = build_retriever(index=index, top_k=5)
    generator_inst = build_generator(request.app.state.config)
    rewriter = build_rewriter(request.app.state.config, generator=generator_inst)
    config = request.app.state.config

    if config.query_rewrite.enabled:
        nodes = retrieve_with_rewriting(retriever, rewriter, QueryBundle(q))
    else:
        nodes = retriever.retrieve(QueryBundle(q))
    sources = _build_sources(nodes)

    return EventSourceResponse(_stream_events(request, generator_inst, nodes, sources, q))


@router.post("/chat/abort")
async def chat_abort(request: Request) -> dict[str, str]:
    for event in request.app.state.cancel_events:
        event.set()
    return {"status": "aborted"}
