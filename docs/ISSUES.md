# Implementation Plan — Vertical Slices

Each issue is a thin, demonstrable end-to-end slice. Work sequentially — each completes before the next begins.

| # | Issue | Type | Blocks | What It Delivers |
|---|-------|------|--------|-----------------|
| 1 | Config-driven pipeline: YAML → Pydantic → Factory | AFK | — | Load `config.yaml`, validate, factory returns stubs. Importable and testable. |
| 2 | PDF ingestion: parse → chunk → embed → store | AFK | #1 | Parse Max Life PDF, chunk, embed with BGE, store in Chroma. `curl /api/health` returns index count. |
| 3 | Retrieval + generation: ask → retrieve → answer | AFK | #2 | REST endpoint answers one question. `curl /api/chat?q=...` returns answer + citations. |
| 4 | Golden Q&A dataset (first 10 pairs) | HITL | #2 | Manually curate 10 QA pairs from PDF into `data/eval/qa.json`. |
| 5 | RAGAS eval harness | AFK | #3, #4 | Run pipeline on QA pairs, compute RAGAS metrics, dump JSON. `python -m src.eval` produces scores. |
| 6 | SSE streaming endpoint | AFK | #3 | Stream tokens via SSE. `curl /api/chat/stream?q=...` shows incremental tokens. |
| 7 | CLI benchmark runner | AFK | #5, #6 | `python -m src.run M0` ingests, evals, saves artifacts to `benchmarks/M0/`. |
| 8 | LangFuse observability | AFK | #7 | `@observe()` decorator on pipeline steps. Traces visible in LangFuse dashboard. |
| 9 | M0 — No-RAG ablation benchmark | AFK | #4, #7 | LLM-only on all eval pairs. Artifacts in `benchmarks/M0/`. |
| 10 | M1 — Naive RAG baseline benchmark | AFK | #9 | Full RAG pipeline on all eval pairs. Artifacts in `benchmarks/M1/`. Compare vs M0. |
| 11 | Web chat UI skeleton | AFK | #6 | React app: text input, SSE streaming, markdown render, disclaimer footer. |
| 12 | System prompt manager | AFK | #3 | Load prompt from `prompts/system_prompt.yaml` by version. Record version in benchmark artifacts. |
| 13 | Config parameter overrides | AFK | #10 | `--chunk-size N` flag for `src.run`. Override any config value from CLI without creating a new YAML per variant. |
| 14 | Execute chunk_size sweep | AFK | #13 | Run 5 benchmarks: chunk_size in [250, 500, 750, 1000, 1500]. Save to `benchmarks/M2{a,b,c,d,e}/`. |
| 15 | M2 comparison report | HITL | #14 | Aggregate 5 results into comparison table. Analyze faithfulness, cost, latency vs chunk size. |
| 16 | Implement remaining chunking strategies | AFK | #15 | Add SemanticChunker, SentenceChunker, agentic variant. Wire into factory dispatch via `chunk.strategy`. |
| 17 | Execute chunking strategy sweep | AFK | #16 | Run 4 benchmarks: recursive, semantic, sentence, agentic. Save to `benchmarks/M3{a,b,c,d}/`. |
| 18 | M3 comparison report | HITL | #17 | Compare chunking strategies. Recommend which to carry forward. |
| 19 | LLM provider interface + implementations | AFK | #18 | `ClaudeGenerator` + `GeminiGenerator` implementing `Generator` ABC. Factory dispatch via `config.llm.model`. `LLMConfig` extended with new model literals. |
| 20 | Execute LLM sweep | AFK | #19 | Run benchmarks comparing GPT-4o-mini, GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash. Save to `benchmarks/M-llm{a,b,c,d}/`. |
| 21 | LLM comparison report | HITL | #20 | Compare cost, faithfulness, latency across 4 providers. Recommend which for accuracy vs budget. |
| 22 | Implement remaining embedding models | AFK | #18 | `OpenAIEmbedder` (text-embedding-3-small), `CohereEmbedder` (cohere-embed-v3), `E5Embedder` (e5-large). Wire into factory dispatch via `embedding.model`. Each wraps its LlamaIndex embedding class. |
| 23 | Execute embedding model sweep | AFK | #22 | Run 3 benchmarks: bge-large, text-embedding-3-small, e5-large. Save to `benchmarks/M4{a,b,d}/`. |
| 24 | M4 comparison report | HITL | #23 | Compare embedding models on faithfulness, recall, cost, latency. |
| 25 | Persistent Chroma + rebuild flag | AFK | #24 | `StorageConfig` with `chroma_path`, `PersistentClient` in factory, lazy-load if collection exists, `--rebuild` CLI flag. |
| 26 | API server loads persisted index | AFK | #25 | `main.py` lifespan uses `build_index(config, force_rebuild=False)` instead of re-ingesting on startup. |
| 27 | Chroma inspect CLI | AFK | #25 | `python -m src.chroma_inspect` dumps all raw chunk text + metadata from the persisted DB. |
| 28 | QueryRewriter ABC + NullQueryRewriter + factory | AFK | #27 | Change `rewrite()` return type to `list[str]`. Implement `NullQueryRewriter` (returns `[query]`). `build_rewriter` dispatch via `query_rewrite.enabled` + `query_rewrite.strategy`. |
| 29 | LLM-based rewriters: HyDE, step-back, multi-query | AFK | #28 | Three classes using the Generator to produce rewritten queries. Each has its own system prompt. HyDE generates hypothetical answer, step-back generates broader question, multi-query generates 3 variants. |
| 30 | Integrate rewriting into retrieval pipeline | AFK | #29 | API routes and eval run rewriter before retriever. Multi-query merges results from all variants with deduplication by node ID. |
| 31 | Execute M5 query rewriting sweep | AFK | #30 | Run 4 benchmarks: no-rewrite vs HyDE vs multi-query vs step-back. Save to `benchmarks/M5{a,b,c,d}/`. |
| 32 | BM25 sparse retriever + node text extraction | AFK | #31 | `BM25Retriever` implementing `Retriever` ABC using `rank_bm25`. Extracts node texts from `VectorStoreIndex` (handles both fresh and persistent loads). Tests with known corpus. |
| 33 | Hybrid retriever (dense + sparse fusion) | AFK | #32 | `HybridRetriever` combining `IndexRetriever` + `BM25Retriever` with reciprocal rank fusion. Uses `retrieval.sparse_weight` / `retrieval.dense_weight` from config. |
| 34 | Update `build_retriever` to dispatch by `retrieval.mode` | AFK | #33 | `build_retriever` accepts config, dispatches to IndexRetriever (dense), BM25Retriever (sparse), or HybridRetriever (hybrid). |
| 35 | Execute M6 retrieval mode sweep | AFK | #34 | Run 3 benchmarks: dense vs sparse vs hybrid. Save to `benchmarks/M6{a,b,c}/`. |
| 36 | Reranker implementations: Cohere, BGE-reranker, cross-encoder | AFK | #35 | Three classes implementing `Reranker` ABC. Cohere uses `co.rerank()` API. BGE-reranker and cross-encoder use `sentence_transformers.CrossEncoder`. `build_reranker` dispatch via `reranker.model`. |
| 37 | RerankingRetriever wrapper + pipeline integration | AFK | #36 | Wraps any retriever with a reranker. When `reranker.enabled`, fetches `reranker.max_input_chunks` (20) nodes, re-ranks to `reranker.top_n` (5). Wired into `build_retriever`. API routes + eval use transparently. |
| 38 | Execute M7 reranker sweep | AFK | #37 | Run 4 benchmarks: no-reranker vs Cohere vs BGE-reranker vs cross-encoder with max_input_chunks=20, top_n=5. Save to `benchmarks/M7{a,b,c,d}/`. |
| 39 | Backend: instrument pipeline steps | AFK | #38 | Each pipeline step (query rewrite → retrieve → rerank → generate) emits SSE events with step name, timing (ms), and cost ($). Refactor `routes.py` streaming endpoint to be step-aware. Add step tracking to every component. |
| 40 | Backend: dev mode API + strategy list | AFK | #39 | `GET /api/strategies` returns available strategies per component (filtered by API keys, implementation status). `GET /api/chat/stream?q=...&mode=dev` accepts per-request strategy overrides (retrieval mode, rewrite, reranker, top-k, LLM model). |
| 41 | Frontend: pipeline visualization component | AFK | #40 | React component showing step-by-step pipeline: each step name, status (pending/running/done), duration bar, cost badge. Updates in real-time via SSE events. Reuses existing streaming infrastructure. |
| 42 | Frontend: strategy override controls | AFK | #41 | Dropdowns for overridable strategies (retrieval mode, rewrite, reranker, top-k, LLM). Read-only info for fixed strategies (chunk strategy, embedding model). Strategy availability fetched from `/api/strategies`. |
| 43 | Frontend: dev mode toggle + RAGAS scores | AFK | #42 | Toggle switch in UI to enable/disable developer mode. When on, shows pipeline visualization + controls + RAGAS scores at end of response. When off, current simple UI (input → answer). |

## Dependency Graph

```
1 ──→ 2 ──→ 3 ──→ 6 ──→ 7 ──→ 8
         │    │            │
         ▼    │            │
         4 ───┘            │
         │                  │
         └──── 5 ──────────┘
                           │
                           9 ──→ 10 ──→ 13 ──→ 14 ──→ 15 ──→ 16 ──→ 17 ──→ 18 ──→ 19 ──→ 20 ──→ 21
                                                                                        │
                                                                                        22 ──→ 23 ──→ 24 ──→ 25 ──→ 26
                                                                                                               │
                                                                                                               27
                                                                                                                │
                                                                                                                 28 ──→ 29 ──→ 30 ──→ 31
                                                                                                                                       │
                                                                                                                                        32 ──→ 33 ──→ 34 ──→ 35
                                                                                                                                                              │
                                                                                                                                                              36 ──→ 37 ──→ 38

3 ──→ 12 (can run parallel with 4-8)

6 ──→ 11 (can start after 6)

11 ──→ 39 ──→ 40 ──→ 41
               │
               42 ──→ 43
```

## Post-M1 Milestones

To be broken into issues after M1 lands:
- M8: Top-k sweep
- M9: Caching
- M10: Best config + GPT-4o
- M11–M17: UX milestones
