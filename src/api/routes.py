from fastapi import APIRouter, Query, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api")


@router.get("/health")
async def health(request: Request) -> dict[str, str | int]:
    index = getattr(request.app.state, "index", None)
    node_count = len(index.index_struct.nodes_dict) if index else 0
    return {"status": "ok", "node_count": node_count}


@router.get("/chat/stream")
async def chat_stream(q: str = Query(..., description="User question")) -> PlainTextResponse:
    # Placeholder: will stream SSE tokens
    return PlainTextResponse(f"Streaming not yet implemented. Query: {q}")


@router.post("/chat/abort")
async def chat_abort() -> dict[str, str]:
    # Placeholder: cancels in-flight generation
    return {"status": "aborted"}
