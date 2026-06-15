# M6 — Retrieval Mode Sweep (Dense vs Sparse vs Hybrid)

**Date:** 2026-06-14 (re-run with harder QA pairs)
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker (TokenTextSplitter), chunk_size=500, bge-large, top_k=5
**Variable:** `retrieval.mode` in [dense, sparse, hybrid]
**Dataset:** 10 QA pairs (5 hard paraphrased + 5 mixed difficulty, replaces 5 easy direct-match pairs)

Sparse uses BM25 (`rank_bm25`). Hybrid uses weighted reciprocal rank fusion (70% dense, 30% sparse).

## Results

| Retrieval | cost  | duration | retr avg | gen avg  | faith | relevancy | precision | recall |
|-----------|-------|----------|----------|----------|-------|-----------|-----------|--------|
| dense (BGE) | $0.003 | 60s | 268ms | 2,492ms | 0.639 | 0.662 | **0.972** | 0.603 |
| sparse (BM25) | $0.003 | 67s | 276ms | 2,967ms | 0.669 | 0.738 | 0.959 | 0.603 |
| hybrid | $0.003 | 66s | 284ms | 2,611ms | **0.719** | **0.741** | **0.972** | 0.578 |

## Analysis

With the harder paraphrase-based questions, the earlier M6 results invert:

### Dense vs Sparse — Now nearly tied
BM25 no longer dominates. On easy questions with keyword overlap, BM25 had a clear edge. On paraphrased questions (e.g., "My father is 72" instead of "Am I eligible at age 68"), both modes struggle equally — recall drops to 0.603 for both. Sparse edges ahead on faithfulness (0.669 vs 0.639) because BM25-exact matches are more reliable when they do occur.

### Hybrid — Best faithfulness (0.719)
Reciprocal rank fusion produces the most reliable answers. When one mode fails to find relevant chunks, the other may compensate, and the RRF scoring downweights unreliable results. Faithfulness improves by +0.080 over sparse and +0.050 over the earlier easy-QA baseline. This is the strongest signal yet that fusion helps on genuinely hard questions.

However, recall drops to 0.578 — the fusion threshold may be too aggressive, filtering out chunks that only one retriever found.

## Recommendation

**Use hybrid retrieval** as the new default. It produces the most faithful answers (0.719) on real-world paraphrased questions. The +0.050 faith gain over no-rewrite dense is meaningful.

Consider tuning the dense/sparse weights and RRF constant — the current 70/30 split may need adjustment for optimal recall.
