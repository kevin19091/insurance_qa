# AGENTS.md

## Project
Insurance Policy QnA Bot — retrieval strategy benchmark. A RAG pipeline that answers insurance policy questions while systematically comparing chunking, embedding, retrieval, and reranking strategies.

## Documentation Map

| Document          | Path           | When to Read                          |
|-------------------|----------------|---------------------------------------|
| PRD               | `docs/PRD.md`   | Understanding goals, architecture, milestones, metrics, and tool choices |
| Infrastructure    | `docs/infra.md` | Infrastructure plan, service map, costs, scaling |
| (future) ADR      | `docs/adr/`     | Architecture Decision Records         |
| (future) Context  | `docs/CONTEXT.md` | Domain model, ubiquitous language |

## Repository Layout

```
data/           Insurance policy PDFs + eval Q&A pairs
docs/           PRD, ADRs, architecture docs
benchmarks/     Per-milestone config + results (config.yaml, eval_results.json, cost_log.json, trace.json)
prompts/        Versioned system prompts
src/            Source code (FastAPI backend, RAG pipeline)
frontend/       React web chat UI
```

## Conventions

- **Config-driven pipeline**: Every RAG component is parameterized via YAML. Swapping strategies = pointing to a different config file. No hardcoded parameters.
- **Milestones are testable strategies**: Each M0–M11 locks all variables except the one being tested. Results go to `benchmarks/Mx/`.
- **Stateless evaluation**: QA is single-turn only. Multi-turn conversation is deferred to future scope.
- **Reproducibility**: Seeds, environment snapshots, and prompt versions are recorded per benchmark run.

## Quick Links for Agents

- **Want to understand what to build?** → `docs/PRD.md`
- **Want to run a milestone?** → `docs/PRD.md` §10, pick a milestone, create `benchmarks/Mx/config.yaml`
- **Want to add a new retrieval strategy?** → Implement the retriever interface, register it in the pipeline factory, create a new config YAML
- **Want to evaluate?** → Run against `data/eval/` golden QA pairs using RAGAS metrics
- **Want to debug a bad answer?** → Search LangFuse trace by query, inspect retrieval → generation chain

## Stack at a Glance

LlamaIndex / FastAPI / React / BGE-large / Chroma→Qdrant / GPT-4o-mini / RAGAS / LangFuse / Unstructured
