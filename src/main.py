"""FastAPI application entry point."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.config import PipelineConfig
from src.observability import get_langfuse
from src.pipeline.factory import build_index

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    get_langfuse()
    config = PipelineConfig.from_yaml(Path("benchmarks/M0/config.yaml"))
    app.state.config = config
    app.state.index = build_index(config)
    app.state.cancel_events = []
    yield


app = FastAPI(title="Insurance QnA Bot", version="0.1.0", lifespan=lifespan)


from src.api.routes import router  # noqa: E402

app.include_router(router)

app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")
