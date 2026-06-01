from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/api")


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/chat/stream")
async def chat_stream(q: str = Query(..., description="User question")) -> PlainTextResponse:
    # Placeholder: will stream SSE tokens
    return PlainTextResponse(f"Streaming not yet implemented. Query: {q}")


@router.post("/chat/abort")
async def chat_abort() -> dict[str, str]:
    # Placeholder: cancels in-flight generation
    return {"status": "aborted"}
