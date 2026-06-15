# M5 — Query Rewriting Strategy Sweep

**Date:** 2026-06-14 (re-run with harder QA pairs)
**Base config:** M1 (Naive RAG) — GPT-4o-mini, recursive chunker (TokenTextSplitter), chunk_size=500, bge-large, dense retrieval top_k=5
**Variable:** `query_rewrite.enabled` + `query_rewrite.strategy` in [none, hyde, multi-query, step-back]
**Dataset:** 10 QA pairs (5 hard paraphrased + 5 mixed difficulty, replaces 5 easy direct-match pairs)

## Results

| Strategy | cost  | duration | retr avg | gen avg  | faith | relevancy | precision | recall |
|----------|-------|----------|----------|----------|-------|-----------|-----------|--------|
| no-rewrite | $0.003 | 75s | 613ms | 2,789ms | **0.724** | 0.662 | 0.969 | 0.603 |
| HyDE | $0.003 | 99s | 3,531ms | 2,781ms | 0.687 | 0.922 | **0.990** | **0.727** |
| multi-query | $0.004 | 112s | 3,331ms | 4,066ms | 0.647 | 0.912 | 0.944 | **0.777** |
| step-back | $0.003 | 91s | 1,668ms | 2,762ms | 0.628 | 0.837 | 0.965 | 0.652 |

*Note: Duration includes RAGAS eval time (~40s). Harder questions reduced relevancy (0.66 vs 0.93 on easy set) because answers are more nuanced.*

## Analysis

### No-rewrite — Best faithfulness (0.724)
Still the strongest for answer accuracy. Dense retrieval with no query rewriting gives the LLM the most reliable context. But recall drops to 0.603 — half the questions fail to retrieve relevant chunks because the paraphrased questions don't share keywords with the source text.

### HyDE — Best precision (0.990), best recall of rewrite strategies (0.727)
Generating a hypothetical answer document bridges the paraphrase gap — it translates "my father is 72" into structured policy language about age eligibility. Precision at 0.990 means almost every retrieved chunk is relevant. However, retrieval latency is 5.7× baseline (3,531ms) due to the extra LLM call.

### Multi-query — Best overall recall (0.777)
Generating 3 query variants captures more coverage but at a cost: lowest faithfulness (0.647) and highest latency/cost. The extra context confuses the LLM.

### Step-back — Weakest (0.628)
Broadening the question loses specificity — "my colleague is turning 70" becomes something generic about eligibility, losing the precise age information needed.

## Recommendation

**Keep no-rewrite as default for answer quality.** Use **HyDE** when recall is critical and you can accept higher latency. Avoid multi-query and step-back for this dataset — they hurt faithfulness without proportionate recall gains.

Compared to the earlier sweep on easy questions, rewriting strategies show their value more clearly on harder questions: HyDE improves recall by +0.124 and precision by +0.021, while nearly matching no-rewrite on faithfulness.
