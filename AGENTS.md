# AGENTS.md

## Project
Insurance Policy QnA Bot — retrieval strategy benchmark. A RAG pipeline that answers insurance policy questions while systematically comparing chunking, embedding, retrieval, and reranking strategies.

## Documentation Map

| Document          | Path           | When to Read                          |
|-------------------|----------------|---------------------------------------|
| PRD               | `docs/PRD.md`   | Understanding goals, architecture, milestones, metrics, and tool choices |
| Infrastructure    | `docs/infra.md` | Infrastructure plan, service map, costs, scaling |
| Architecture      | `docs/ARCHITECTURE.md` | Component diagrams, data flow, config-driven pipeline, request lifecycle |
| ADR               | `docs/adr/`     | Architecture Decision Records (why LlamaIndex, why SSE, why PyMuPDF) |
| Context           | `docs/CONTEXT.md` | Domain model, ubiquitous language, term glossary |
| Issues            | `docs/ISSUES.md` | Implementation plan — vertical slices in dependency order |

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
- **Want to see implementation order?** → `docs/ISSUES.md`
- **Want to run a milestone?** → `docs/PRD.md` §10, pick a milestone, create `benchmarks/Mx/config.yaml`. First run ingests into persistent Chroma (`data/chroma/`), subsequent runs skip re-ingestion.
- **Want to force re-ingestion?** → `python -m src.run M1 --rebuild` or `rm -rf data/chroma/`
- **Want to inspect raw vector DB entries?** → `python -m src.chroma_inspect`
- **Want to add a new retrieval strategy?** → Implement the retriever interface, register it in the pipeline factory, create a new config YAML
- **Want to evaluate?** → Run against `data/eval/` golden QA pairs using RAGAS metrics
- **Want to debug a bad answer?** → Search LangFuse trace by query, inspect retrieval → generation chain

## Stack at a Glance

LlamaIndex / FastAPI / React / BGE-large / Chroma→Qdrant / GPT-4o-mini / RAGAS / LangFuse / Unstructured

## Development Workflow

Every feature or benchmark follows this cycle in order:

1. **`/grill-with-docs`** — Before writing code or creating issues, grill the plan against `docs/CONTEXT.md` (domain language), `docs/adr/` (decisions), `docs/ARCHITECTURE.md` (data flow). Sharpen terminology, catch contradictions, update docs inline as decisions crystallise.

2. **`/to-issues`** — Break the approved plan into vertical-slice issues. Each issue is a thin end-to-end path through *all* layers (schema → API → UI → tests), NOT a horizontal slice of one layer. Prefer AFK (automated) over HITL (human-in-the-loop) where possible. Append to `docs/ISSUES.md`.

3. **`/tdd`** — Implement each issue test-first (red → green → refactor). Write the test that defines success, then make it pass, then clean up. Tests that require API keys or model loading are guarded with `@pytest.mark.skipif` or `@pytest.mark.slow` — never skip writing them.

4. **`/improve-codebase-architecture`** (or manual refactor) — After each issue lands, review for consolidation, dead code, tightly-coupled modules, or AI-unfriendly patterns. Refactor in tiny independent commits, never mixed with feature work.

Exceptions: Trivial bugfixes (one-line, no design question) may skip #1–#2. Everything else runs the full cycle. Never move to the next issue without explicit approval.
