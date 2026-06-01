"""API routes for chat and health."""

from typing import cast

from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse
from llama_index.core.schema import QueryBundle, TextNode

from src.pipeline.factory import build_generator, build_retriever

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request) -> dict[str, str | int]:
    index = getattr(request.app.state, "index", None)
    node_count = len(index.index_struct.nodes_dict) if index else 0
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


@router.get("/chat/stream")
async def chat_stream(q: str = Query(..., description="User question")) -> PlainTextResponse:
    return PlainTextResponse(f"Streaming not yet implemented. Query: {q}")


@router.post("/chat/abort")
async def chat_abort() -> dict[str, str]:
    return {"status": "aborted"}
