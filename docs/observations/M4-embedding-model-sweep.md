# M4 — Embedding Model Sweep

**Date:** 2026-06-12
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker (TokenTextSplitter), chunk_size=500, dense retrieval top_k=5
**Variable:** `embedding.model` in [bge-large, text-embedding-3-small, e5-large]
**Skipped:** cohere-embed-v3 (no `COHERE_API_KEY` set)

## Results

| Model | nodes | cost  | duration | retr avg | gen avg  | faith | relevancy | precision | recall |
|-------|-------|-------|----------|----------|----------|-------|-----------|-----------|--------|
| bge-large | 37 | $0.003 | 134s | 305ms | 2,637ms | 0.787 | 0.932 | 0.918 | 0.725 |
| text-embedding-3-small | 37 | $0.003 | 48s | 400ms | 2,396ms | 0.664 | 0.824 | 0.934 | 0.717 |
| e5-large | 37 | $0.003 | 240s | 232ms | 2,833ms | 0.673 | 0.926 | 0.909 | 0.785 |

All models produced the same number of nodes (37) — chunking is independent of embedding. Differences are purely in retrieval quality.

## Analysis

### bge-large — Best all-around
Highest faithfulness (0.787) and relevancy (0.932). Moderate recall (0.725). Balanced across all metrics. The default choice remains the best.

### text-embedding-3-small — Fastest indexing (48s vs 134s+)
OpenAI's API-based embedding is 3× faster than local BGE and 5× faster than e5-large. But faithfulness (0.664) and relevancy (0.824) are notably lower — the smaller embedding space (1536-dim but used at default) loses semantic fidelity for this domain.

### e5-large — Best recall (0.785), slowest indexing (240s)
Leads in recall but trails in faithfulness (0.673). Takes 5× longer than OpenAI and 2× longer than BGE to index. The additional retrieval coverage doesn't translate to better answers.

## Recommendation

**Keep bge-large as the default.** It has the best balance of faithfulness, speed, and cost. The local model is free (no API calls), fast enough (134s), and produces the highest-quality answers.

Consider **text-embedding-3-small** only if indexing speed is critical and you can accept a ~0.12 drop in faithfulness.

## Next

Swap Chroma to persistent storage — index survives restarts, no re-ingesting between runs.
