# M3 — Chunking Strategy Comparison

**Date:** 2026-06-12
**Base config:** M1 (Naive RAG) — GPT-4o-mini, BGE-large, dense retrieval top_k=5, chunk_size=250
**Variable:** `chunk.strategy` in [recursive, sentence, semantic, agentic]
**Fixed:** chunk_size=250, chunk_overlap=50, 10 eval QA pairs

## Results

| Strategy | nodes | cost  | faith | relevancy | precision | recall | gen avg | total avg |
|----------|-------|-------|-------|-----------|-----------|--------|---------|-----------|
| recursive | 76  | $0.002 | 0.692 | 0.924 | 0.960 | 0.877 | 2,858ms | 3,188ms |
| sentence  | 77  | $0.002 | 0.672 | 0.940 | 0.972 | 0.765 | 2,699ms | 2,934ms |
| semantic  | 19  | $0.006 | 0.720 | 0.928 | 0.944 | 0.955 | 4,614ms | 4,817ms |
| agentic   | 293 | $0.001 | 0.749 | 0.732 | 0.914 | 0.617 | 3,405ms | 3,627ms |

## Analysis

### Recursive (TokenTextSplitter)
Splits on whitespace/tokens with no awareness of sentence boundaries. Produces 76 chunks — clean, predictable size. Best all-around: second-best recall (0.877), good generation speed (2.9s), moderate cost ($0.002).

### Sentence (SentenceSplitter)
Sentence-aware splitting. Produces 77 chunks — similar volume to recursive. Highest precision (0.972) and relevancy (0.940), but lowest recall (0.765) among the fast strategies. The sentence boundary constraint creates chunks that may miss cross-sentence context.

### Semantic (SemanticSplitterNodeParser)
Uses BGE-large embeddings to find natural breakpoints at semantic boundaries. Only 19 chunks — very coarse granularity. Highest recall (0.955) because each large chunk contains more relevant text, but generation is slowest (4.6s) due to long context prompts. 3× the cost of recursive. Faithfulness (0.720) is second-best — fewer, denser chunks provide coherent context.

### Agentic (SemanticDoubleMergingSplitterNodeParser)
Splits then merges chunks based on semantic similarity. Produces 293 chunks — extremely fine granularity. Lowest recall (0.617) and relevancy (0.732) — too many tiny chunks fragment relevant information across retrieval units. Cheapest ($0.001) and decent generation speed (3.4s) because each chunk is small. Highest faithfulness (0.749) — small focused chunks reduce hallucination risk.

## Recommendation

**recursive** is the best default for general use: strong recall (0.877), fast generation (2.9s), low cost ($0.002).

**semantic** is best when recall is critical (e.g., fact lookup) — accepts 2× cost and 1.7× latency for 0.955 recall.

**agentic** produces too many fragments — not recommended without a higher top-k to compensate.

Carry forward **recursive** as the default for M4+.
