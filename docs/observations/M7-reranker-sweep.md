# M7 — Reranker Sweep (top-20 → rank-5)

**Date:** 2026-06-14
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker, chunk_size=500, bge-large, dense retrieval
**Variable:** `reranker.enabled` + `reranker.model` in [none, bge-reranker, cross-encoder]
**Reranker config:** top_n=5, max_input_chunks=20
**Dataset:** 10 QA pairs (5 hard paraphrased + 5 mixed difficulty)

Skipped: Cohere (no `COHERE_API_KEY` set).

## Results

| Reranker | cost  | duration | retr avg | gen avg | faith | relevancy | precision | recall |
|----------|-------|----------|----------|---------|-------|-----------|-----------|--------|
| no-reranker | $0.003 | 59s | 311ms | 2,300ms | **0.730** | 0.747 | 0.972 | **0.603** |
| bge-reranker | $0.003 | 61s | 397ms | 2,679ms | 0.634 | 0.661 | **1.000** | **0.603** |
| cross-encoder | $0.003 | 54s | 291ms | 2,156ms | 0.624 | 0.661 | 0.900 | 0.583 |

## Analysis

### No-reranker — Best overall (0.730 faith)
Retrieving top-5 directly from dense search produces the most faithful answers. The rerankers filter out context that the LLM could use, hurting answer quality.

### BGE-reranker — Perfect precision (1.000) but faith drops (−0.096)
BGE-reranker achieves perfect precision — every retrieved chunk is relevant. But it achieves this by being extremely selective, filtering out useful context. Faithfulness drops 0.096 (0.730 → 0.634) because the LLM has less information to work with.

### Cross-encoder — Underperforms
Lower precision (0.900), lower recall (0.583), and lowest faithfulness (0.624). The smaller MiniLM model is too aggressive or noisy for this domain.

## Recommendation

**Skip reranking for this dataset.** With only 37 nodes and top-5 retrieval, the dense retriever already returns high-quality results. Rerankers hurt faithfulness without improving recall. 

Rerankers are more valuable when:
- Retrieving from much larger corpora (1000+ nodes)
- Initial retrieval has low precision (many false positives)
- Using a higher max_input_chunks (50+)

Cohere reranker may perform differently — test when `COHERE_API_KEY` is available.
