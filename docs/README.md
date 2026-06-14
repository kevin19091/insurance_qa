# Insurance QnA Bot — Code Overview

## What this is

A FastAPI server that answers insurance policy questions using RAG (Retrieval-Augmented Generation). It ingests a PDF, chunks it, embeds it into a vector store, then answers questions by retrieving relevant chunks and feeding them to an LLM.

The whole point of this repo is **benchmarking**: systematically swapping one strategy variable at a time (chunk size, chunking method, embedding model, LLM provider, etc.) and comparing the RAGAS metrics, cost, and latency.

---

## Project Structure

```
insurance_qa/
├── data/
│   ├── chroma/                    # Persistent Chroma SQLite DB
│   ├── max-life-....pdf          # The source policy document
│   └── eval/
│       └── qa.json               # 10 golden Q&A pairs (the eval set)
├── benchmarks/
│   ├── M0/                       # No-RAG ablation results
│   ├── M1/                       # Naive RAG baseline results
│   ├── M2a/ ... M2e/             # Chunk size sweep (250-1500)
│   ├── M3a/ ... M3d/             # Chunking strategy comparison
│   ├── M4a/ M4b/ M4d/            # Embedding model sweep
│   └── each has:
│       ├── config.yaml           # The config used for this run
│       ├── trace.json            # Timing, node count, overrides
│       ├── cost_log.json         # Token counts + estimated USD
│       └── eval_results.json     # RAGAS scores + per-question answers
├── src/
│   ├── main.py                   # FastAPI app entry point + lifespan
│   ├── config.py                 # Pydantic models for pipeline config
│   ├── eval.py                   # Eval harness (run pipeline + RAGAS)
│   ├── run.py                    # CLI benchmark runner
│   ├── chroma_inspect.py         # CLI to inspect raw Chroma DB entries
│   ├── observability.py          # LangFuse decorator re-export
│   ├── api/
│   │   └── routes.py             # /api/chat, /api/chat/stream, /api/health, /api/chat/abort
│   └── pipeline/
│       ├── __init__.py           # ABC interfaces for each component
│       ├── parser.py             # PyMuPDFParser
│       ├── chunker.py            # RecursiveChunker, SentenceChunker, SemanticChunker, AgenticChunker
│       ├── embedder.py           # BgeEmbedder, OpenAIEmbedder, CohereEmbedder, E5Embedder
│       ├── retriever.py          # IndexRetriever, NullRetriever
│       ├── generator.py          # OpenAIGenerator, ClaudeGenerator, GeminiGenerator
│       ├── factory.py            # Builds components from PipelineConfig
│       ├── reranker.py           # Stub (not yet implemented)
│       └── rewriter.py           # Stub (not yet implemented)
├── frontend/                     # React chat UI (Vite)
│   ├── src/App.jsx               # Main chat component
│   └── package.json
├── tests/
│   ├── test_config.py            # PipelineConfig + factory dispatch tests
│   ├── test_ingestion.py         # Chunker, embedder, index, persistent Chroma tests
│   ├── test_chat.py              # Retriever, generator, chat/stream/abort endpoints
│   ├── test_chroma_inspect.py    # Chroma inspect CLI tests
│   ├── test_eval.py              # QA pair loading, run_eval structure
│   ├── test_run.py               # Cost estimate, usage log, override parsing, benchmarks, rebuild flag
│   ├── test_generator.py         # Prompt building, usage tracking, factory dispatch
│   ├── test_observability.py     # LangFuse observe decorator
│   └── e2e/chat_ui.mjs           # Playwright E2E test (headless Chromium)
├── docs/
│   ├── PRD.md                    # Product requirements
│   ├── ARCHITECTURE.md           # Component diagrams, data flow
│   ├── CONTEXT.md                # Domain glossary
│   ├── ISSUES.md                 # Implementation plan (vertical slices)
│   ├── observations/             # Benchmark analysis reports
│   │   ├── M2-chunk-size-sweep.md
│   │   ├── M3-chunking-strategy-comparison.md
│   │   └── M4-embedding-model-sweep.md
│   └── adr/                      # Architecture Decision Records
├── AGENTS.md                     # Agent routing + dev workflow
└── pyproject.toml                # Python dependencies
```

---

## How It Works

### The Request Lifecycle

```
User question
    │
    ▼
GET /api/chat?q=...
    │
    ├── 1. Retrieve: Vector search in Chroma → top-k chunks with scores
    │
    ├── 2. Generate: LLM (GPT / Claude / Gemini) answers with citations
    │
    └── 3. Response: { "answer": "...", "sources": [...pages+scores] }
```

The streaming version (`/api/chat/stream`) sends SSE events:
```
event: sources   → {"sources": [{"page": 8, "text": "...", "score": 0.58}, ...]}
event: token     → {"token": "The"}
event: token     → {"token": " premium"}
...              → ...
event: done      → [DONE]
```

### The Pipeline Components

Every component follows the same pattern:
1. An **ABC** in `src/pipeline/__init__.py` defines the interface
2. **Concrete implementations** live in their own file
3. A **factory function** in `factory.py` builds the right implementation from config

| Component | Interface | Implementations | Config field |
|-----------|-----------|-----------------|-------------|
| Parser | `Parser.parse()` | `PyMuPDFParser` | — |
| Chunker | `Chunker.chunk()` | `RecursiveChunker`, `SentenceChunker`, `SemanticChunker`, `AgenticChunker` | `chunk.strategy` |
| Embedder | `Embedder.embed()` | `BgeEmbedder`, `OpenAIEmbedder`, `E5Embedder` | `embedding.model` |
| Retriever | `Retriever.retrieve()` | `IndexRetriever`, `NullRetriever` | `retrieval.top_k` |
| Generator | `Generator.generate()` / `.stream()` | `OpenAIGenerator`, `ClaudeGenerator`, `GeminiGenerator` | `llm.model` |
| Reranker | `Reranker.rerank()` | *(stub)* | `reranker.enabled` |
| QueryRewriter | `QueryRewriter.rewrite()` | *(stub)* | `query_rewrite.enabled` |
| Storage | — | Chroma (SQLite, persistent) | `storage.chroma_path` |

### Configuration

Everything is parameterized via YAML + Pydantic:

```yaml
# benchmarks/M1/config.yaml
chunk:
  strategy: "recursive"
  chunk_size: 500
  chunk_overlap: 50
retrieval:
  top_k: 5
llm:
  model: "gpt-4o-mini"
storage:
  chroma_path: "data/chroma"       # Persistent vector DB location
```

Override from CLI without editing files:
```bash
python -m src.run M1 --override chunk.chunk_size=250 --override llm.model=claude-3.5-sonnet
```

---

## How to Run Things

### Start the server

```bash
source .venv/bin/activate
python -m uvicorn src.main:app --port 8000
# → http://localhost:8000 (React UI) or /api/chat?q=...
```

The index is **persistent** — first start ingests the PDF into Chroma at `data/chroma/`. Subsequent starts load the existing index (no re-ingestion).

### Run a benchmark

```bash
python -m src.run M0                    # Full benchmark (ingest + eval)
python -m src.run M1 --skip-eval        # Ingest only
python -m src.run M1 --rebuild          # Force re-ingestion (ignore cached index)
python -m src.run M1 --override chunk.chunk_size=250
```

Results go to `benchmarks/<milestone>/` with `trace.json`, `cost_log.json`, `eval_results.json`.

### Inspect the vector DB

```bash
python -m src.chroma_inspect            # Dump all raw entries from data/chroma/
python -m src.chroma_inspect --path my/chroma/path
```

### Run tests

```bash
# Fast tests (no LLM calls, no model loading)
python -m pytest tests/test_config.py tests/test_generator.py tests/test_observability.py

# Slow tests (load BGE model, call LLMs)
python -m pytest tests/ -m "not slow"

# E2E (requires Playwright)
node tests/e2e/chat_ui.mjs
```

### Run the eval harness directly

```bash
python -m src.eval --config benchmarks/M0/config.yaml
```

---

## Testing — What the Tests Cover

| Test file | What it tests |
|-----------|---------------|
| `test_config.py` | Config defaults, YAML loading, validation errors, factory dispatch for chunker strategies, parser returns documents |
| `test_ingestion.py` | All 4 chunker types produce nodes, BGE embedder outputs correct dimensions, index is queryable, health endpoint returns node count, persistent Chroma lazy-loads and rebuilds |
| `test_chat.py` | Retriever returns scored nodes, NullRetriever returns empty, generator produces answer from context, streaming SSE endpoint returns token/done events, abort endpoint works |
| `test_chroma_inspect.py` | Inspect CLI dumps raw entries with correct schema, CLI round-trip via `--path` flag |
| `test_eval.py` | QA pair loading (10 pairs, required fields), run_eval returns correct structure and scores |
| `test_run.py` | Cost estimation, usage log building, config override parsing and type coercion, applying overrides to config, benchmark produces artifacts, `--rebuild` flag forces re-ingestion |
| `test_generator.py` | Prompt building (with context, without context, multiple contexts, source citations), usage starts at zero, factory dispatch for OpenAI/Claude/Gemini models |
| `test_observability.py` | Observe decorator works without LangFuse config, get_langfuse returns a client |
| `e2e/chat_ui.mjs` | Full end-to-end: starts server, opens Chromium, types question, asserts streaming response contains expected content |

### Test conventions

- Tests that need `OPENAI_API_KEY` are guarded with `@pytest.mark.skipif(not _OPENAI_AVAILABLE, ...)`
- Tests that load BGE (slow) are marked `@pytest.mark.slow`
- Gemini tests are guarded with `GOOGLE_API_KEY`

---

## Adding a New Strategy

Adding a new embedding model, chunking method, or LLM provider follows the same recipe:

1. **Create a class** implementing the ABC (e.g., `class CohereEmbedder(EmbedderABC)`)
2. **Add it to the factory** — one `if` branch in `build_embedder()` 
3. **Add the model name** to the `Literal` type in `config.py`
4. **Write tests** — factory dispatch returns the right type, basic operation works
5. **Run the benchmark** — `python -m src.run M1 --override embedding.model=cohere-embed-v3`

That's it. The CLI override system means you never need to create a new YAML file for a single experiment.

---

## LLM Providers

The generator factory dispatches on the `llm.model` prefix:

| Prefix | Generator class | Requires |
|--------|----------------|----------|
| `gpt-*` | `OpenAIGenerator` | `OPENAI_API_KEY` |
| `claude-*` | `ClaudeGenerator` | `ANTHROPIC_API_KEY` |
| `gemini-*` | `GeminiGenerator` | `GOOGLE_API_KEY` |

Example:
```bash
python -m src.run M1 --override llm.model=claude-3.5-sonnet
```

---

## Observability (LangFuse)

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`, every pipeline step is traced:

```
Parser.parse      → span
RecursiveChunker  → span
BgeEmbedder       → embedding
IndexRetriever    → retriever
OpenAIGenerator   → generation
```

View traces at your LangFuse project dashboard. No-op when not configured.

---

## Key Files for an Onboarding Deep-Dive

| If you want to understand... | Read |
|----------------------------|------|
| The big picture + milestones | `docs/PRD.md` |
| Data flow through the pipeline | `src/pipeline/factory.py` (the `build_index` function) |
| How config drives everything | `src/config.py` |
| How benchmarks run end-to-end | `src/run.py` |
| The API surface | `src/api/routes.py` |
| The React chat UI | `frontend/src/App.jsx` |
| How new issues are structured | `docs/ISSUES.md` |
| Previous benchmark findings | `docs/observations/` |
