# Infrastructure Plan

## 1. Summary

Single-VM architecture: one Railway (or Fly.io) instance runs the FastAPI backend, serves the React frontend as static files, hosts LangFuse for observability, and caches everything in-process. Zero AWS, zero separate frontend host.

## 2. Service Map

| Service              | Provider           | Plan / Tier                     | Monthly Cost |
|----------------------|--------------------|----------------------------------|--------------|
| Backend + Frontend   | Railway            | Hobby ($5/mo) or Fly.io free    | $0–5         |
| LLM API              | OpenAI (GPT-4o-mini)| Pay-per-token                   | $3–15 (dev)  |
| LLM Benchmark        | OpenAI (GPT-4o)    | Pay-per-token, on-demand only   | ~$2/run      |
| Vector DB (M1–M5)   | In-process Chroma (persistent SQLite) | None (runs on VM)               | $0           |
| Vector DB (M6+)      | Qdrant Cloud       | Free tier (1GB cluster)         | $0           |
| Observability        | LangFuse (self-hosted) | Docker on same VM           | $0           |
| Embedding            | BGE-large (CPU)    | Runs on VM, no GPU              | $0           |
| Caching              | `functools.lru_cache` | In-process                   | $0           |
| PDF Storage          | Local disk         | VM ephemeral storage            | $0           |
| **Total (dev)**      |                    |                                  | **$3–15/mo** |
| **Total (active users)** |                |                                  | **$50–80/mo** |

## 3. Single-VM Layout

```
┌─────────────────────────────────────────────┐
│  Railway / Fly.io VM (2 vCPU, 4GB RAM)       │
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ uvicorn   │  │ Chroma   │  │ LangFuse    │ │
│  │ FastAPI   │  │ (in-proc)│  │ (Docker)   │ │
│  │  :8000    │  │          │  │  :3000      │ │
│  └─────┬─────┘  └──────────┘  └────────────┘ │
│        │                                       │
│        │  /api/*  →  API endpoints             │
│        │  /*       →  React SPA (static files) │
│        │                                       │
│  ┌─────┴─────┐  ┌───────────┐                 │
│  │ LRU Cache  │  │ PDFs on   │                 │
│  │ in-memory  │  │ disk      │                 │
│  └───────────┘  └───────────┘                 │
│                                               │
│              │                                 │
└──────────────┼─────────────────────────────────┘
               │
        OpenAI API (external)
```

## 4. Streaming

SSE (Server-Sent Events) over plain HTTP. React frontend consumes via native `EventSource`.

| Endpoint            | Purpose              |
|---------------------|----------------------|
| `GET /api/chat/stream?q=...` | Stream answer tokens|
| `POST /api/chat/abort`       | Cancel in-flight generation |

SSE chosen over WebSocket because:
- Single-direction stream (no bidirectional need).
- Simpler connection lifecycle (no heartbeat, no upgrade handshake).
- Abort handled by a separate POST endpoint — clean enough for this scale.

## 5. Why Not Vercel

React build is served from FastAPI via `StaticFiles`. No separate frontend host. No API keys shipped to the browser. All secrets stay on the backend VM.

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")
```

## 6. No GPU Strategy

BGE-large-en runs on CPU (~2GB RAM). Embedding happens at ingest time (once per PDF upload), not at query time. Query embedding is negligible. This avoids GPU provisioning entirely.

## 7. Scaling Notes

| Threshold                          | Action                                   |
|------------------------------------|------------------------------------------|
| >50 concurrent streaming sessions  | Move to a dedicated server or scale replicas |
| PDF corpus >10 documents           | Add S3 bucket for document storage       |
| Cache memory pressure              | Offload to Redis (Railway has managed Redis) |
| User growth >100 DAU               | Add authentication, rate limiting        |

## 8. Environment Variables

```
OPENAI_API_KEY=sk-...
# BGE runs locally, no key needed
# Chroma runs in-process, no key needed
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
# Qdrant (M6+)
QDRANT_URL=https://xxx.qdrant.cloud
QDRANT_API_KEY=...
```
