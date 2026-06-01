# ARCHITECTURE.md

## Overview

The Insurance QnA Bot is a **config-driven RAG pipeline**. Every component (chunker, embedder, retriever, reranker, generator, rewriter) is parameterized via YAML. Swapping strategies = pointing to a different `config.yaml`. No hardcoded parameters.

## Component Diagram

```
                           ┌──────────────────────────────┐
                           │        LangFuse Tracing        │
                           │    (every function wrapped     │
                           │     with @observe())           │
                           └──────────────┬─────────────────┘
                                          │
   ┌──────────────────────────────────────┼──────────────────────────────────────┐
   │                                      ▼                                       │
   │  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
   │  │   Parser     │───▶│   Chunker    │───▶│   Embedder   │───▶│  VectorDB  │ │
   │  │ (pdf→Docs)   │    │ (Docs→Nodes) │    │ (Nodes→Vecs) │    │  (Chroma)  │ │
   │  └─────────────┘    └──────────────┘    └──────────────┘    └─────┬──────┘ │
   │                                                                    │        │
   │                               INGESTION (batch, once per PDF)       │        │
   └────────────────────────────────────────────────────────────────────┼────────┘
                                                                         │
   ┌─────────────────────────────────────────────────────────────────────┼────────┐
   │                               QUERY (per-user-request)              │        │
   │                                                                     │        │
   │  ┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌────────▼──────┐ │
   │  │    User      │   │    Query     │   │  Retriever  │   │   VectorDB    │ │
   │  │   Question   │──▶│   Rewriter   │──▶│ (dense/bm25 │──▶│  (Chroma/     │ │
   │  │              │   │  (optional)  │   │  /hybrid)   │   │   Qdrant)    │ │
   │  └──────────────┘   └──────────────┘   └──────┬──────┘   └──────────────┘ │
   │                                                │                           │
   │       ┌────────────────────────────────────────┘                           │
   │       ▼                                                                    │
   │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
   │  │   Reranker   │──▶│   Generator  │──▶│  Guardrails  │──▶│  Answer    │ │
   │  │  (optional)  │   │   (LLM)      │   │ (scope check │   │  + Cites   │ │
   │  │   top-20→N   │   │              │   │ +disclaimer) │   │            │ │
   │  └──────────────┘   └──────────────┘   └──────────────┘   └────────────┘ │
   │                                                                     │    │
   └─────────────────────────────────────────────────────────────────────┼────┘
                                                                          │
                                                                    SSE Stream
                                                                   to Frontend
```

## Data Flow

### 1. Ingestion (offline, once per PDF)
```
PDF file
  → Parser.parse(file_path) → list[Document]        # Raw text + metadata
  → Chunker.chunk(docs)     → list[Document]         # Split into nodes
  → Embedder.embed(nodes)   → list[Document]         # Add embedding vectors
  → VectorDB.add(nodes)                               # Persist in Chroma
```

### 2. Query (online, per request)
```
User question (string)
  → QueryRewriter.rewrite(query)   → str              # Optional: HyDE / multi-query
  → Retriever.retrieve(QueryBundle) → list[NodeWithScore]  # Dense / sparse / hybrid
  → Reranker.rerank(query, nodes)   → list[NodeWithScore]  # Optional re-rank
  → Generator.stream(query, nodes)  → SSE tokens            # LLM text generation
  → Guardrails.check(answer)        → (pass/fail)           # Scope + disclaimer
  → Frontend renders answer + citations
```

## Config-Driven Pipeline

All strategy parameters live in `benchmarks/Mx/config.yaml`:

```yaml
chunk:
  strategy: recursive        # recursive | semantic | sentence | agentic
  chunk_size: 500
  chunk_overlap: 50

embedding:
  model: bge-large           # bge-large | text-embedding-3-small | cohere-embed-v3
  dimension: 1024

retrieval:
  mode: dense                # dense | sparse | hybrid
  top_k: 5

reranker:
  enabled: false

query_rewrite:
  enabled: false

llm:
  model: gpt-4o-mini
  temperature: 0.0

seed: 42
prompt_version: v1
```

The `PipelineFactory` reads this config and returns the corresponding component instances. Adding a new strategy (e.g., "agentic" chunking) means:
1. Implement `Chunker` interface in `src/pipeline/chunker.py`
2. Register it in the factory's `build_chunker()` dispatch
3. Set `chunk.strategy: agentic` in config

## Request Lifecycle (SSE)

```
1. Browser → GET /api/chat/stream?q=Is cardiac surgery covered?
2. FastAPI route handler:
   a. QueryRewriter.rewrite(q)           [optional]
   b. Retriever.retrieve(QueryBundle)
   c. Reranker.rerank(q, nodes)          [optional]
   d. Generator.stream(q, nodes)         → yields tokens via SSE
   e. Guardrails applied to final text
3. Browser renders tokens as they arrive via EventSource
4. If user clicks "Stop" → Browser POSTs /api/chat/abort
   → Backend cancels the generator task
```

## Directory Map

```
src/pipeline/__init__.py   ← Interface contracts (ABCs)
src/pipeline/parser.py     ← PDF parser implementations
src/pipeline/chunker.py    ← Chunking strategy implementations
src/pipeline/embedder.py   ← Embedding model implementations
src/pipeline/retriever.py  ← Retrieval mode implementations
src/pipeline/reranker.py   ← Reranker implementations
src/pipeline/rewriter.py   ← Query rewriting implementations
src/pipeline/generator.py  ← LLM generator implementations
src/pipeline/factory.py    ← build_*() dispatch from config
src/config.py              ← Pydantic PipelineConfig model
src/api/routes.py          ← FastAPI endpoints
src/main.py                ← FastAPI app + static file mount
src/eval/metrics.py        ← RAGAS evaluation runner
benchmarks/Mx/config.yaml  ← Per-milestone strategy config
prompts/system_prompt.yaml ← Versioned system prompts
```

## Key Constraints

- **Stateless queries**: No multi-turn conversation in M0–M10. Each question is independent.
- **Deterministic where possible**: Seeds recorded in config. LLM temperature = 0 for eval.
- **Reproducible**: Environment snapshot (`pip freeze`) recorded per benchmark run.
- **Single VM**: Backend, Chroma, LangFuse all run on one Railway/Fly.io instance.
