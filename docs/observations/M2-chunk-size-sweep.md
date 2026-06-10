# M2 — Chunk Size Sweep

**Date:** 2026-06-02
**Base config:** M1 (Naive RAG) — GPT-4o-mini, BGE-large, dense retrieval top_k=5, no reranker
**Variable:** `chunk_size` in [250, 500, 750, 1000, 1500]
**Fixed:** chunk_overlap=50, SentenceSplitter, Chroma (ephemeral), 10 eval QA pairs

## Results

| chunk_size | nodes | cost  | tokens | retr avg | gen avg  | total avg | faith | relevancy | precision | recall |
|-----------|-------|-------|--------|----------|----------|-----------|-------|-----------|-----------|--------|
| 250       | 77    | $0.002 | 10,978 | 227ms   | 3,198ms | 3,425ms  | 0.742 | 0.938     | 0.952     | 0.765  |
| 500       | 37    | $0.003 | 20,597 | 223ms   | 3,547ms | 3,770ms  | 0.742 | 0.936     | 0.939     | 0.708  |
| 750       | 25    | $0.005 | 30,455 | 220ms   | 6,132ms | 6,352ms  | 0.762 | 0.831     | 0.917     | 0.917  |
| 1000      | 21    | $0.006 | 36,153 | 235ms   | 3,823ms | 4,058ms  | 0.675 | 0.837     | 0.961     | 0.955  |
| 1500      | 20    | $0.006 | 36,163 | 225ms   | 4,368ms | 4,593ms  | 0.795 | 0.924     | 0.944     | 0.955  |

## Analysis

### Faithfulness
Faithfulness hovers around 0.74 for 250–750, dips to 0.675 at 1000, then recovers to 0.795 at 1500. The dip at 1000 suggests chunks are too large to pinpoint the relevant sentence but not large enough to provide sufficient surrounding context. The 1500 recovery indicates that with very large chunks, the relevant passage is always included even if buried in noise.

### Recall
Steady improvement from 0.708 (500) to 0.955 (1000+). Larger chunks increase the probability that the relevant passage appears in the top-5 retrieved chunks. The jump from 500→750 is the most dramatic (+0.21).

### Cost & Latency
- 250 is cheapest ($0.002, 3.4s total) — fewer tokens per prompt
- 750 is an outlier in generation latency (6.1s) despite not having the most tokens — possibly a RAGAS eval timing artifact
- Beyond 1000, diminishing returns: cost plateaus at $0.006, latency stabilizes around 4–4.6s

### Precision
Stays high across all sizes (0.917–0.961). Retrieval precision is not sensitive to chunk size in this range.

## Recommendation

**chunk_size = 250** offers the best tradeoff:

| Why | Value |
|-----|-------|
| Lowest cost | $0.002 per run |
| Fastest generation | 3.2s avg |
| Tied best faithfulness | 0.742 (tied with 500) |
| Best relevancy | 0.938 |
| Good recall | 0.765 (only 250 beats 500) |

If recall is the priority (e.g., for fact-heavy lookup questions), consider **chunk_size = 750** which delivers 0.917 recall at moderate cost. However, the generation latency spike at 750 should be investigated — it may be a one-off anomaly.

## Next

Carry forward **chunk_size=250** into M3 (chunking strategy comparison) as the fixed baseline.
