# M6 — Retrieval Mode Sweep (Dense vs Sparse vs Hybrid)

**Date:** 2026-06-14
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker (TokenTextSplitter), chunk_size=500, bge-large, top_k=5
**Variable:** `retrieval.mode` in [dense, sparse, hybrid]

Sparse uses BM25 (`rank_bm25`). Hybrid uses weighted reciprocal rank fusion (70% dense, 30% sparse).

## Results

| Retrieval | cost  | duration | retr avg | gen avg  | faith | relevancy | precision | recall |
|-----------|-------|----------|----------|----------|-------|-----------|-----------|--------|
| dense (BGE) | $0.003 | 80s | 246ms | 2,766ms | 0.756 | 0.931 | 0.866 | 0.758 |
| sparse (BM25) | $0.003 | 73s | 242ms | 2,768ms | 0.756 | 0.927 | **0.918** | **0.792** |
| hybrid | $0.003 | 82s | 249ms | 3,336ms | 0.748 | 0.927 | 0.913 | 0.705 |

## Analysis

### Sparse (BM25) — Surprising leader
BM25 keyword retrieval matches or beats dense across every metric:
- Same faithfulness (0.756) and virtually identical cost/latency
- Higher precision (0.918 vs 0.866) — BM25 finds more on-topic chunks
- Higher recall (0.792 vs 0.758) — BM25 casts a wider keyword net that still lands relevant results

Insurance policy text uses precise, domain-specific terminology ("cardiac surgery", "pre-existing condition", "premium payment") — exactly the kind of language BM25 excels at matching.

### Dense (BGE) — Solid baseline
Consistent performance across all metrics. Retrieval latency is low (246ms). The embedding captures semantic relationships but doesn't improve over exact keyword matching for this dataset.

### Hybrid — Lower recall (0.705)
Weighted fusion underperforms both individual modes. The RRF combination seems to dilute strong single-mode signals. Possible improvements:
- Tune dense/sparse weights (currently 70/30)
- Use a cutoff before fusion (only include nodes both retrievers agree on)

## Recommendation

**Use sparse (BM25) as the default.** It's simpler (no embedding needed for retrieval), faster, and produces better recall and precision on this insurance Q&A dataset. Keep dense as a fallback for questions requiring semantic understanding rather than keyword matches.

**Skip hybrid** unless the weights are tuned or the fusion strategy is improved.
