# PRD: Insurance Policy QnA Bot — Retrieval Strategy Benchmark

## 1. Problem Statement
Customers of Max Life Insurance need quick answers about their Group Credit Life Secure policy—coverage, exclusions, claims, definitions, and payments—without reading a dense legal PDF. The broader goal of this repo is to **systematically benchmark different retrieval strategies** (chunking, embedding, retrieval, reranking) on a real-world insurance QnA task and track their performance across multiple metrics.

## 2. Target Audience
- **Primary**: Policyholders and customers of Max Life Insurance.
- **Secondary**: Engineers/researchers evaluating RAG retrieval strategies on this dataset.

## 3. Goals
- Build a QnA bot that answers insurance policy questions using RAG.
- Provide concise natural-language summaries with citations (section/page references from the PDF).
- Gracefully reject out-of-scope questions with a clear "I don't know" response.
- Support the full range of question types:
  - **Coverage** — "Is cardiac surgery covered?"
  - **Claims** — "How do I file a claim?"
  - **Exclusions** — "What is NOT covered?"
  - **Definitions** — "What does 'pre-existing condition' mean?"
  - **Premiums/Payments** — "When is the premium due?"
  - **General** — Any question answerable from the policy document.
- **Benchmark different retrieval strategies** — each milestone is one testable strategy variant.
- Produce comparable metric snapshots across strategies to drive architectural decisions.

## 4. Non-Goals
- Answering questions outside the scope of the uploaded policy document (handled via guardrails/refusal).
- Handling non-insurance documents (e.g., medical reports, claim forms).
- Real-time policy updates via customer uploads—new documents require re-ingestion.
- Multi-language support (initial release: English only).
- Production deployment to end users (focus is benchmarking, not shipping).

## 5. Technical Stack

| Layer              | Technology                                  | Variants to Test                          |
|--------------------|---------------------------------------------|-------------------------------------------|
| Frontend           | React (web chat UI)                         | —                                         |
| Backend            | FastAPI (Python)                            | —                                         |
| RAG Framework      | LlamaIndex                                  | —                                         |
| PDF Parser         | Unstructured / PyMuPDF / LlamaParse         | Multiple parsers                          |
| Chunking           | RecursiveCharacter / Semantic / Agentic     | Chunk size, overlap, strategy             |
| Embedding          | BGE-large / OpenAI / Cohere / E5            | Dense only, sparse-only, hybrid           |
| Vector DB          | Chroma / Qdrant                             | —                                         |
| Retrieval          | Dense-only / Sparse-only / Hybrid           | Top-k, score threshold                    |
| Query Rewriting    | LLM-based / HyDE / Multi-query              | Rewrite strategies                        |
| Reranking          | Cohere / BGE-reranker / cross-encoder       | With and without reranker                 |
| LLM                | GPT-4o-mini (eval) / GPT-4o (benchmark)     | —                                         |
| Guardrails         | Custom prompt / Guardrails-AI               | —                                         |
| Observability      | LangFuse (tracing, cost, latency per query) | —                                         |
| Eval Framework     | RAGAS                                       | —                                         |
| Caching            | Embedding cache + LLM response cache        | With and without cache                    |
| Infra              | Docker + local-first                        | —                                         |

### Tool Choice Rationale

| Choice           | Why This Over Alternatives                                                                 |
|------------------|-------------------------------------------------------------------------------------------|
| **LlamaIndex** over LangChain | Purpose-built for document QnA. Clean `IngestionPipeline` → `VectorStoreIndex` → `QueryEngine` separation. Chunking/retrieval strategy swapping is a one-liner config change. LangChain requires more glue code for the same modularity. |
| **Unstructured** over PyMuPDF | Insurance PDFs are table-heavy. Unstructured preserves table structure and metadata boundaries. PyMuPDF is faster but loses table semantics. |
| **BGE-large-en** as primary embedding | Open source (MIT), top MTEB scores, zero API cost — more experiments possible. OpenAI/Cohere as baseline comparisons. |
| **Chroma** → **Qdrant**  | Chroma for M1–M5 (simple, local-first). Switch to Qdrant at M7 when hybrid retrieval needs native dense+sparse fusion. |
| **RAGAS** over DeepEval | Purpose-built for RAG metrics (faithfulness, context precision/recall, answer relevancy). DeepEval is a broader eval framework less optimized for retrieval evaluation. |
| **LangFuse** over LangSmith | Open source (MIT), self-hostable, RAG-native tracing. One decorator `@observe()` captures retrieval context, LLM I/O, tokens, latency, cost. No vendor lock-in. |
| **GPT-4o-mini** (eval) + **GPT-4o** (benchmark) | 4o-mini is cheap enough to run all 30–50 QA pairs per milestone without breaking budget. 4o used for final quality benchmarks to measure ceiling. |

## 6. Architecture (High-Level)

```
                         ┌─────────────────────┐
                         │   LangFuse Tracing   │ ← observability sidecar
                         └──────┬──────────────┘
                                │
   User Query ──→ Query Rewriter ──→ Retriever ──→ Reranker (opt) ──→ LLM Generator ──→ Guardrails ──→ Answer
                     │                    │                                    │                │
                     │              ┌─────┴──────┐                      ┌──────┴──────┐    ┌────┴──────────┐
                     │          Dense  BM25  Hybrid                  Prompt Mgmt    │  Refusal check │
                     │                                        (versioned YAML)       │  Disclaimer    │
                     │                                                                └───────────────┘
                     │
               Config YAML ──→ Pipeline Factory (swaps strategies per milestone)
```

**Key properties:**
- **Config-driven**: Every pipeline component is specified in a YAML file. Swapping strategies = pointing to a different config.
- **Strategy-as-plugin**: Chunking, embedding, retrieval, and reranking are each abstracted behind an interface. New strategies added without touching pipeline code.
- **LangFuse sidecar**: Every function call in the pipeline is wrapped with `@observe()` for automatic tracing.

## 7. Data Source
- **Single source now**: `max-life-group-credit-life-secure-policy-document-v1.pdf`
- **Future**: Multiple PDFs — ingestion layer must support adding documents without code changes.
- **Eval dataset**: Curated golden Q&A pairs (30–50) extracted from the PDF, stored in `data/eval/`. Eval set is versioned alongside code.

## 8. Constraints & Risks

| Constraint / Risk                                    | Mitigation                                                      |
|------------------------------------------------------|-----------------------------------------------------------------|
| PDF layout (tables, fine print) may lose structure   | Test multiple PDF parsers; use chunking that preserves tables.  |
| LLM hallucination on policy specifics                | Cite source chunks; LLM-as-judge faithfulness eval; guardrails. |
| Single PDF today, many tomorrow                      | Abstract loader/chunker interfaces from day one.                |
| No existing QA dataset to evaluate                   | Curate 30–50 golden Q&A pairs from the PDF.                    |
| Benchmarking requires disciplined isolation          | Each milestone locks all variables except the one being tested. |
| Regulatory liability from incorrect insurance advice | Mandatory disclaimer on every answer: "This is AI-generated and not legal advice. Refer to the official policy document." |
| Non-reproducible benchmark results                   | Seed management, environment pinning (requirements.txt / poetry.lock), config versioning in each benchmark run. |

## 9. Success Metrics
For every milestone, measure and report:

### Retrieval Metrics
| Metric                     | Definition                                       | Target         |
|----------------------------|--------------------------------------------------|----------------|
| **Context Precision**      | Are retrieved chunks all relevant?               | ≥80%           |
| **Context Recall**         | Are all relevant chunks retrieved?               | ≥80%           |
| **Hit Rate @ k**           | Is the golden chunk in top-k results?            | ≥90% @ k=5     |
| **MRR**                    | Mean Reciprocal Rank of golden chunk             | ≥0.85          |

### Generation Metrics
| Metric                     | Definition                                       | Target         |
|----------------------------|--------------------------------------------------|----------------|
| **Answer Correctness**     | LLM-as-judge / human eval of factual accuracy    | ≥90%           |
| **Faithfulness**           | Does the answer contradict the source chunks?    | ≥95%           |
| **Answer Relevancy**       | Is the answer on-topic for the question?         | ≥85%           |

### Safety & Scope Metrics
| Metric                     | Definition                                       | Target         |
|----------------------------|--------------------------------------------------|----------------|
| **Refusal Rate**           | Correctly rejects out-of-scope questions         | ≥90%           |
| **Noise Robustness**       | Does retrieval survive misspelling/noisy input?  | Tracked        |
| **Retrieval Value**        | Improvement over no-RAG ablation baseline        | Tracked        |

### Performance & Cost Metrics
| Metric                     | Definition                                       | Target         |
|----------------------------|--------------------------------------------------|----------------|
| **Time to First Token**    | Time from request to first output token          | ≤2s            |
| **End-to-End Latency**     | Total request → response time                    | ≤5s            |
| **Cost per Query**         | API cost (embed + LLM + reranker) per question   | Tracked        |
| **Cache Hit Rate**         | % queries served from cache vs. fresh            | Tracked        |

### UX & Usability Metrics
| Metric                     | Definition                                       | Target         |
|----------------------------|--------------------------------------------------|----------------|
| **Feedback Rate**          | % answers receiving thumbs up/down               | Tracked        |
| **Thumbs Up Ratio**        | % of feedback that is positive                   | ≥80%           |
| **Autocomplete Usage**     | % queries triggered from suggestion vs. typed    | Tracked        |
| **Bookmark Rate**          | % answers bookmarked by users                    | Tracked        |
| **Session Return Rate**    | % users resuming a previous chat session         | Tracked        |

All metrics are logged per milestone in `benchmarks/<milestone>/` for comparison.

## 10. Milestones (Testable Strategies)

Each milestone locks all knobs except the one being tested. Results from all prior milestones are archived for cross-comparison.

| Milestone | Strategy Variant                                                   | What It Tests                         |
|-----------|--------------------------------------------------------------------|---------------------------------------|
| **M0**    | No-RAG: LLM-only answering from training data (GPT-4o, GPT-4o-mini)| Retrieval ablation baseline           |
| **M1**    | Naive RAG: fixed chunk 500/50, BGE-large, dense top-5, GPT-4o-mini | End-to-end plumbing + baseline        |
| **M2**    | Chunk size sweep (250, 500, 1000, 1500, 2000)                     | Optimal chunk size                    |
| **M3**    | Chunking strategy: semantic vs. recursive vs. sentence-split vs. agentic | Best chunking approach           |
| **M4**    | Embedding model sweep (BGE-large, text-embedding-3-small, Cohere embed-v3, E5) | Best embedding model      |
| **M5**    | Query rewriting: no-rewrite vs. HyDE vs. multi-query vs. step-back | Rewriting impact on retrieval         |
| **M6**    | Sparse retrieval (BM25) vs. dense (BGE) vs. hybrid (dense+sparse)  | Retrieval paradigm                    |
| **M7**    | Reranker on top-20 (Cohere / BGE-reranker / cross-encoder)         | Reranker impact                       |
| **M8**    | Top-k sweep (3, 5, 10, 20, 30) with best config                    | Optimal retrieval depth               |
| **M9**    | Caching: embedding cache + LLM response cache enabled               | Cache impact on latency/cost          |
| **M10**   | Best config assembled into final RAG pipeline + GPT-4o             | Crown-jewel comparison vs. M0 and M1  |
| **M11**   | Web chat UI basics (input, response, markdown render, disclaimer)  | Frontend skeleton + safety + E2E      |
| **M12**   | Chat history persistence (localStorage / backend DB)               | Session management, resume, navigate  |
| **M13**   | Bookmarked answers (save/unsave Q&A pairs)                         | Persistent bookmarking for revisit    |
| **M14**   | Starter suggestions + autocomplete suggestions                      | Discovery + input quality             |
| **M15**   | User feedback (thumbs up/down + reason prompt)                     | Feedback loop for eval improvement    |
| **M16**   | Streaming responses (SSE token-by-token)                           | Perceived latency, abort mid-stream   |
| **M17**   | Share & export (copy, shareable link, PDF export)                  | Utility for sharing with family/agent |

### Milestone Artifacts
Each milestone directory (`benchmarks/Mx/`) contains:
- `config.yaml` — locked parameters for this run (chunk size, embedding model, retrieval mode, top-k, LLM, seeds)
- `eval_results.json` — aggregate metrics (all metrics from Section 9)
- `per_question_metrics.csv` — row per Q&A pair with individual scores
- `cost_log.json` — API cost breakdown (embedding + LLM + reranker)
- `trace.json` — LangFuse trace export for the run
- `environment.txt` — `pip freeze` / `poetry lock` snapshot for reproducibility

## 11. Reproducibility
- **Seed management**: Every pipeline component (chunking, embedding, LLM) uses a configurable random seed recorded in `config.yaml`.
- **Environment pinning**: `requirements.txt` or `poetry.lock` committed per benchmark run.
- **Deterministic pipeline**: All non-LLM components produce identical output given the same seed and config.
- **Versioned eval set**: Golden Q&A pairs in `data/eval/` are versioned. Any addition/modification to the eval set triggers a full re-benchmark.

## 12. Safety & Guardrails
- **Scope boundary**: Questions are classified as in-scope vs. out-of-scope. Out-of-scope → polite refusal with suggestion to contact human support.
- **Confidence threshold**: If retrieved chunk similarity scores are below threshold, the bot refuses and says "I couldn't find relevant information in the policy document."
- **Mandatory disclaimer**: Every answer includes *"This response is AI-generated and does not constitute legal or insurance advice. Please refer to the official policy document for complete and binding terms."*
- **UI disclaimer**: Chat interface shows permanent disclaimer footer.

## 13. Prompt Management
- System prompt is versioned in `prompts/system_prompt.yaml`.
- Prompt includes: role definition, citation format, refusal criteria, disclaimer requirement, tone guidelines.
- Each benchmark run records the prompt version in `config.yaml`.
- Prompt ablations (different citation styles, different tone) are valid future milestones.

## 14. Observability (LangFuse)
- **Function-level tracing**: `@observe()` decorator on every pipeline step (parse, chunk, embed, retrieve, rerank, generate).
- **Per-query trace**: Input query → rewritten query → retrieved chunks → reranked chunks → LLM input/output → guardrail check → final answer.
- **Auto-captured**: Token counts, latency per step, cost per query, embedding model used.
- **Debugging workflow**: User reports a bad answer → search LangFuse by query → inspect trace to see if retrieval failed or generation failed → fix the right component.
- **Dashboard**: Aggregate latency histograms, cost trends, retrieval precision trends across milestone runs.

## 15. Future Scope
- **Multi-document ingestion** — support uploading/updating policy documents dynamically.
- **Multi-turn conversation memory** — handle follow-up questions within a session.
- **Advanced chunking** — agentic chunking, hierarchical chunking, table-aware splitting.
- **Re-ranking ensembles** — cascade multiple rerankers.
- **SLM distillation** — replace GPT-4 with a distilled model to reduce cost.
- **Online A/B testing** — compare strategies on real user queries.
- **Multi-language support** — Hindi, other Indian languages for broader accessibility.
- **Citation highlight in PDF viewer** — click a citation to open the PDF scrolled to the exact paragraph.
- **Voice input** — speech-to-text for mobile/accessibility.
