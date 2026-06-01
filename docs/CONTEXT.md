# CONTEXT.md — Domain Language

Canonical terms for all contributors. When a term appears in code, docs, or conversation, it means exactly what's defined here.

## RAG Pipeline

| Term | Definition |
|------|------------|
| **Document** | A `llama_index.core.schema.Document` — represents the raw text extracted from one PDF page (or section) before chunking. Has `.text` and `.metadata` (page number, source filename). |
| **Node** | A `llama_index.core.schema.Document` after chunking. Represents one retrieval unit. Has `.text`, `.metadata`, and optionally `.embedding`. |
| **Chunk** | Synonym for Node in casual usage. A segment of text returned by the Chunker. |
| **Chunking Strategy** | The algorithm that splits Documents into Nodes. Values: `recursive` (character-based recursive split), `semantic` (split on semantic boundaries via LLM/sentence embeddings), `sentence` (split on sentence boundaries), `agentic` (LLM decides split points). |
| **Embedding Model** | The model that converts text to a dense vector. Values: `bge-large`, `text-embedding-3-small`, `cohere-embed-v3`, `e5-large`. |
| **VectorDB** | The store that indexes embeddings and supports similarity search. Values: Chroma (M1–M5), Qdrant (M6+). |
| **Retrieval Mode** | How documents are fetched. Values: `dense` (vector similarity), `sparse` (BM25 keyword), `hybrid` (dense + sparse fusion). |
| **Retriever** | The component that takes a `QueryBundle` and returns `list[NodeWithScore]`. |
| **NodeWithScore** | A `llama_index.core.schema.NodeWithScore` — a Node plus a relevance score (0–1). |
| **Top-k** | Number of nodes returned by the Retriever before reranking (e.g., top_k: 20). |
| **Reranker** | A cross-encoder or ranking model that re-orders retrieved nodes for relevance. Optional, runs after Retriever. Values: `cohere`, `bge-reranker`, `cross-encoder`. |
| **Query Rewriter** | Transforms a raw user question into a retrieval-optimized form. Values: `hyde` (generate hypothetical answer then embed), `multi-query` (generate multiple variants), `step-back` (generate broader context question). |
| **Generator** | The LLM that produces the final answer from context nodes. |
| **Guardrails** | Post-generation checks: in-scope classification, mandatory disclaimer injection, confidence threshold. |
| **Citation** | A reference to the source document section included in the LLM answer. Format: `[Source: Section X]`. |

## Eval & Benchmarking

| Term | Definition |
|------|------------|
| **Golden Q&A** | A manually curated question–answer pair from the policy PDF. Stored in `data/eval/`. Used to score retrieval and generation quality. |
| **Milestone** | A single experiment (M0–M17) that tests one variable while locking all others. Results go to `benchmarks/Mx/`. |
| **Ablation** | Removing one component to measure its impact. M0 is the no-RAG ablation (LLM-only). |
| **RAGAS** | The evaluation framework (`ragas`) used to compute metrics: faithfulness, answer relevancy, context precision, context recall. |
| **Hit Rate @ k** | Fraction of queries where the correct chunk appears in the top-k retrieved results. |
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank for the first correct chunk across all queries. |
| **TTFT** (Time to First Token) | Milliseconds between request and the first SSE token. |
| **Latency** (End-to-End) | Milliseconds between request and the final SSE marker. |

## Infrastructure

| Term | Definition |
|------|------------|
| **LangFuse** | Open-source observability tool. Traces every pipeline step, captures tokens, latency, cost. Self-hosted on the same VM. |
| **SSE** (Server-Sent Events) | HTTP protocol for server-to-client streaming. Used to stream LLM tokens to the browser. `/api/chat/stream` returns `text/event-stream`. |
| **StaticFiles mount** | FastAPI serves the React SPA from `frontend/build/` at the root path `/*`. No separate frontend host. |

## Insurance Domain

| Term | Definition |
|------|------------|
| **Policy Document** | The official PDF from Max Life Insurance describing coverage, exclusions, claims process, definitions, and terms. |
| **Coverage** | What medical events/conditions are financially covered by the policy. |
| **Exclusion** | What is explicitly NOT covered. |
| **Claim** | A formal request by the policyholder for payment under the policy. |
| **Premium** | The payment made by the policyholder to maintain coverage. |
