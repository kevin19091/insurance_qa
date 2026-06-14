# M5 — Query Rewriting Strategy Sweep

**Date:** 2026-06-14
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker (TokenTextSplitter), chunk_size=500, bge-large, dense retrieval top_k=5
**Variable:** `query_rewrite.enabled` + `query_rewrite.strategy` in [none, hyde, multi-query, step-back]

## Results

| Strategy | cost  | duration | retr avg | gen avg  | faith | relevancy | precision | recall |
|----------|-------|----------|----------|----------|-------|-----------|-----------|--------|
| no-rewrite | $0.003 | 85s | 613ms | 3,614ms | 0.775 | 0.930 | 0.913 | 0.733 |
| HyDE | $0.003 | 109s | 3,198ms | 3,261ms | 0.722 | 0.927 | **0.962** | 0.738 |
| multi-query | $0.004 | 115s | 2,790ms | 4,597ms | 0.714 | 0.918 | 0.902 | **0.855** |
| step-back | $0.003 | **78s** | 1,176ms | 3,187ms | **0.774** | **0.933** | 0.882 | 0.688 |

*Note: Duration includes RAGAS eval time (~40s), not just retrieval+generation.*

## Analysis

### No-rewrite — Best all-around
Faithfulness (0.775) and relevancy (0.930) are the highest or tied. Lowest retrieval latency (613ms). No extra LLM cost for rewriting. The baseline remains the strongest overall choice.

### HyDE — Best precision (0.962)
Generates a hypothetical answer document before retrieval, which boosts context precision. But:
- Retrieval latency jumps 5× (613ms → 3,198ms) — each query does one extra LLM call
- Faithfulness drops to 0.722 — the hypothetical answer may bias the generation
- Recall barely changes (0.733 → 0.738)

### Multi-query — Best recall (0.855)
Generates 3 query variants, retrieves each, merges results. Recall jumps from 0.733 to 0.855. But:
- Cost 33% higher ($0.004 vs $0.003) — extra LLM calls per query
- Retrieval latency 4.5× baseline (2,790ms)
- Faithfulness drops to 0.714 — more context doesn't mean better answers

### Step-back — Fastest total time (78s), tied faithfulness (0.774)
Generates a broader question but retrieval is only 2× slower than baseline (1,176ms). However recall drops to 0.688 — the broader question loses specificity.

## Recommendation

**Keep no-rewrite as default.** Rewriting adds latency and cost without improving faithfulness. The extra retrieval coverage from multi-query (recall +0.122) doesn't translate to better answers (faith -0.061).

**Use multi-query only if recall is the priority** and you can accept the extra cost and latency. All rewriting strategies hurt faithfulness — they introduce context that the LLM may misinterpret.
