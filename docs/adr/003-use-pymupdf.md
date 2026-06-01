# ADR 003: Use PyMuPDF over Unstructured

**Status**: Accepted  
**Date**: 2026-06-01

## Context

We need a PDF parser that extracts text from insurance policy documents (tables, fine print, multi-column layouts). `unstructured` has better table preservation but has a build issue on macOS (depends on `llvmlite`/`numba` which fails to compile from source on Python 3.12).

## Decision

Use **PyMuPDF** (fitz) for now. Revisit `unstructured` if table extraction becomes a bottleneck.

## Rationale

1. **Pre-built wheels**: PyMuPDF ships binary wheels for all platforms including macOS ARM. No compilation required.
2. **Fast text extraction**: PyMuPDF is one of the fastest PDF text extractors.
3. **Table-aware chunking can compensate**: If tables are lost during text extraction, semantic chunking strategies (M3) can help re-group related content.
4. **Lower operational risk**: Adding a dependency that requires source compilation (unstructured) introduces a fragile build step. PyMuPDF Just Works.

## Consequences

- Some table structure may be lost during extraction
- If insurance PDFs are heavily tabular, may need to revisit this decision at M3 (chunking strategy milestone)
- The `Parser` interface abstracts the choice — swapping later is a config change + implementing one new class
