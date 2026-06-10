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
| 18 | M3 comparison report | HITL | #17 | Compare chunking strategies. Recommend which to carry forward.

## Dependency Graph

```
1 ──→ 2 ──→ 3 ──→ 6 ──→ 7 ──→ 8
         │    │            │
         ▼    │            │
         4 ───┘            │
         │                  │
         └──── 5 ──────────┘
                           │
                           9 ──→ 10 ──→ 13 ──→ 14 ──→ 15 ──→ 16 ──→ 17 ──→ 18

3 ──→ 12 (can run parallel with 4-8)

6 ──→ 11 (can start after 6)
```

## Post-M1 Milestones

To be broken into issues after M1 lands:
- M2: Chunk size sweep
- M3: Chunking strategy comparison
- M4: Embedding model sweep
- M5: Query rewriting strategies
- M6: Dense vs. sparse vs. hybrid retrieval
- M7: Reranker impact
- M8: Top-k sweep
- M9: Caching
- M10: Best config + GPT-4o
- M11–M17: UX milestones
