"""API routes for chat and health."""

import asyncio
import json
import time
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


def _step_event(step: str, status: str, duration_ms: float = 0, cost_usd: float = 0) -> dict[str, str]:
    return {
        "event": "step",
        "data": json.dumps({
            "step": step,
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "cost_usd": cost_usd,
        }),
    }


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    COST_PER_1K: dict[str, float] = {
        "gpt-4o-mini": 0.000150,
        "gpt-4o": 0.0025,
        "claude-3.5-sonnet": 0.003,
        "claude-3-opus": 0.015,
        "gemini-2.0-flash": 0.0001,
    }
    rate = COST_PER_1K.get(model, 0.000150)
    return ((prompt_tokens / 1000) * rate) + ((completion_tokens / 1000) * rate * 2)


@router.get("/chat")
async def chat(
    request: Request,
    q: str = Query(..., description="User question"),
    mode: str = Query("normal", description="'dev' for pipeline trace"),
) -> dict[str, Any]:
    index = getattr(request.app.state, "index", None)
    if index is None:
        return {"answer": "Index not available. Please ingest a document first.", "sources": []}

    config = request.app.state.config
    retriever = build_retriever(index=index, top_k=5, config=config)
    generator = build_generator(config)
    rewriter = build_rewriter(config, generator=generator)

    pipeline_trace: list[dict[str, Any]] = []

    t0 = time.time()
    if config.query_rewrite.enabled:
        nodes = retrieve_with_rewriting(retriever, rewriter, QueryBundle(q))
    else:
        nodes = retriever.retrieve(QueryBundle(q))
    retrieval_ms = (time.time() - t0) * 1000
    pipeline_trace.append({"step": "retrieve", "duration_ms": round(retrieval_ms, 1), "cost_usd": 0.0})

    t0 = time.time()
    answer = generator.generate(q, nodes)
    gen_ms = (time.time() - t0) * 1000
    gen_cost = _estimate_cost(
        config.llm.model,
        generator.usage.get("prompt_tokens", 0),
        generator.usage.get("completion_tokens", 0),
    )
    pipeline_trace.append({"step": "generate", "duration_ms": round(gen_ms, 1), "cost_usd": gen_cost})

    sources = [
        {
            "page": n.node.metadata.get("page"),
            "text": cast(TextNode, n.node).text[:300],
            "score": round(n.score, 3) if n.score else None,
        }
        for n in nodes
    ]

    result: dict[str, Any] = {"answer": answer, "sources": sources}
    if mode == "dev":
        result["pipeline_trace"] = pipeline_trace
    return result


def _build_sources(nodes: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "page": n.node.metadata.get("page"),
            "text": cast(TextNode, n.node).text[:300],
            "score": round(n.score, 3) if n.score else None,
        }
        for n in nodes
    ]


async def _stream_events_with_dev(
    request: Request,
    generator_inst: Any,
    retriever: Any,
    rewriter: Any,
    q: str,
    config: Any,
    dev_mode: bool,
) -> AsyncGenerator[dict[str, str], None]:
    cancel_event = asyncio.Event()
    request.app.state.cancel_events.append(cancel_event)
    try:
        steps: list[dict[str, Any]] = []

        if dev_mode:
            yield _step_event("retrieve", "start")
        t0 = time.time()
        if config.query_rewrite.enabled:
            actual_nodes = retrieve_with_rewriting(retriever, rewriter, QueryBundle(q))
        else:
            actual_nodes = retriever.retrieve(QueryBundle(q))
        retrieval_ms = (time.time() - t0) * 1000
        if dev_mode:
            yield _step_event("retrieve", "complete", retrieval_ms)
            steps.append({"step": "retrieve", "duration_ms": round(retrieval_ms, 1), "cost_usd": 0.0})

        sources_data = _build_sources(actual_nodes)
        yield {"event": "sources", "data": json.dumps({"sources": sources_data})}

        if dev_mode:
            yield _step_event("generate", "start")
        t0 = time.time()
        async for token in generator_inst.stream(q, actual_nodes):
            if cancel_event.is_set():
                yield {"event": "aborted", "data": json.dumps({"status": "aborted"})}
                return
            yield {"event": "token", "data": json.dumps({"token": token})}
        gen_ms = (time.time() - t0) * 1000
        gen_cost = _estimate_cost(
            config.llm.model,
            generator_inst.usage.get("prompt_tokens", 0),
            generator_inst.usage.get("completion_tokens", 0),
        )
        if dev_mode:
            yield _step_event("generate", "complete", gen_ms, gen_cost)
            steps.append({"step": "generate", "duration_ms": round(gen_ms, 1), "cost_usd": gen_cost})

        yield {"event": "done", "data": "[DONE]"}
        if dev_mode and steps:
            yield {"event": "pipeline_trace", "data": json.dumps({"steps": steps})}
    finally:
        with suppress(ValueError):
            request.app.state.cancel_events.remove(cancel_event)


async def _error_stream(msg: str) -> AsyncGenerator[dict[str, str], None]:
    yield {"event": "error", "data": json.dumps({"error": msg})}


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    q: str = Query(..., description="User question"),
    mode: str = Query("normal", description="'dev' for step-by-step trace"),
) -> EventSourceResponse:
    index = getattr(request.app.state, "index", None)
    if index is None:
        return EventSourceResponse(_error_stream("No index available"))

    config = request.app.state.config
    retriever = build_retriever(index=index, top_k=5)
    generator_inst = build_generator(config)
    rewriter = build_rewriter(config, generator=generator_inst)
    dev_mode = mode == "dev"

    return EventSourceResponse(
        _stream_events_with_dev(request, generator_inst, retriever, rewriter, q, config, dev_mode)
    )


@router.post("/chat/abort")
async def chat_abort(request: Request) -> dict[str, str]:
    for event in request.app.state.cancel_events:
        event.set()
    return {"status": "aborted"}
