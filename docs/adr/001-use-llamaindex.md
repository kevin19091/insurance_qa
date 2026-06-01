# ADR 001: Use LlamaIndex over LangChain

**Status**: Accepted  
**Date**: 2026-06-01

## Context

The project needs a RAG framework to parse PDFs, chunk documents, embed text, retrieve nodes, and generate answers. Two dominant Python options: LangChain and LlamaIndex.

## Decision

Use **LlamaIndex**.

## Rationale

1. **Purpose-built for document QnA**: LlamaIndex's `IngestionPipeline` → `VectorStoreIndex` → `QueryEngine` primitives map directly to our pipeline stages. LangChain's LCEL (LangChain Expression Language) requires more glue code for the same modularity.
2. **Cleaner component separation**: Swapping a chunking strategy in LlamaIndex is a `NodeParser` substitution. In LangChain it requires re-wiring the `DocumentTransformer` chain. Our project's core requirement is easy strategy swapping.
3. **Better ingestion DSL**: LlamaIndex's `SimpleDirectoryReader`, `IngestionPipeline`, and `NodeParser` abstract away PDF parsing boilerplate. LangChain's `DocumentLoaders` are more general but require more configuration for this use case.
4. **Community momentum**: LlamaIndex adoption is growing faster in the RAG space specifically. LangChain is broader (agent orchestration, tool use) which we don't need.

## Consequences

- Must use `llama-index` ecosystem packages (`llama-index-embeddings-*`, `llama-index-llms-*`, etc.)
- LangFuse integration works via LlamaIndex's callback system
- Generator interface wraps `llama_index` but keeps our own ABC for strategy abstraction
