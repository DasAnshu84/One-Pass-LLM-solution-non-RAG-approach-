# Structured Fact-Table Retrieval — POC README

## Problem

Standard RAG (chunk → embed → vector top-k) has three known weaknesses on a
multi-hundred-document corpus of policy/procurement text:

1. **Cross-document / whole-corpus questions** — answers that require combining
   facts from several documents are missed, because top-k similarity surfaces
   *locally similar* chunks, not an *exhaustive* set of relevant facts.
2. **Index maintenance** — every doc change means re-chunking and re-embedding,
   and there's no clean way to update just one document's contribution.
3. **Near-misses** — semantically close but wrong chunks get retrieved instead
   of the actually relevant (but differently worded) clause.

Constraint: no embedding-based vector search as the *core* retrieval
mechanism, and per-query cost/context must stay in the same range as a RAG
baseline (a handful of short passages, not the whole corpus).

## Approach

Replace the vector index with a **structured fact table**, built once
offline, queried cheaply at runtime.

### Steps

1. **Offline extraction (build-time, done once per doc)**
   For each document, one LLM call extracts every distinct rule / condition /
   threshold / authority statement into structured rows:
   `{rule, source_doc, source_clause}`. This is the direct analog of
   "embedding a chunk" in RAG — same cost shape, done once, re-run only for
   docs that change.

2. **Fact table (the replacement for the vector index)**
   All extracted rows across all docs form a single table. It is small,
   structured, and auditable — each fact is traceable to an exact document
   and clause.

3. **Query-time lookup (bounded context, RAG-comparable cost)**
   At query time, filter the fact table down to the relevant rows (currently:
   pass the whole table for small corpora; at scale, filter via
   keyword/BM25/entity match — see Tradeoffs). Feed only the filtered rows to
   the LLM to produce the final answer with citations.

4. **Baseline for comparison (naive RAG)**
   A standard chunk → TF-IDF "vector" → cosine top-k → answer pipeline is kept
   alongside, purely to demonstrate where it misses facts that the structured
   approach catches — not as part of the proposed system.

## Why this addresses the original weaknesses

- **Cross-document questions**: every doc's facts are present in the table by
  construction, so a question needing Doc A + B + C just needs all three rows
  retrieved — no similarity ranking to "lose."
- **Maintenance**: updating one document means re-extracting that one doc's
  rows, not rebuilding an index over the whole corpus.
- **Near-misses**: facts are short, self-contained, and exact-match/keyword
  filterable — less prone to "semantically close but wrong" retrieval than
  chunk embeddings.

## Tradeoffs

- **Extraction cost and quality become the new bottleneck.** Hundreds of docs
  means hundreds of LLM calls upfront (offline, parallelizable, one-time —
  but not free). Bad extraction silently breaks downstream answers, the same
  way bad chunking silently breaks RAG.
- **The fact table doesn't scale to "pass it all in" past a few dozen docs.**
  At hundreds of docs (potentially thousands of fact rows), the table no
  longer fits in one prompt. A filtering step (BM25/keyword/entity match) is
  required before the final LLM call — this POC's "pass the whole table"
  shortcut only works at small scale.
- **Loses some nuance.** Structured one-sentence rules can flatten
  caveats, cross-references, or ambiguous phrasing that's only fully
  expressible in the original prose. Raw-passage retrieval (e.g. BM25 over
  chunks) is a useful fallback for supporting evidence/quotes.
- **No semantic generalization.** Without embeddings, exact wording mismatches
  (synonyms, paraphrases) can still cause misses in the filtering step —
  pure keyword/BM25 has the opposite failure mode of vector search.

## Later enhancements

- **Add a filter layer before the final LLM call**: SQLite-backed fact table
  + BM25 (or entity-tag matching) to retrieve the top ~30-50 relevant rows
  per query instead of the full table, restoring RAG-comparable per-query
  cost at corpus scale.
- **Query routing**: classify each question (single-fact / cross-document /
  whole-corpus aggregation) with one small LLM call, and route aggregation
  questions to a direct SQL/pandas query over the fact table instead of an
  LLM context-stuffing call — this is what makes "across the whole corpus"
  questions cheap rather than proportional to corpus size.
- **Entity/topic tagging during extraction** to enable fast exact-match
  pre-filtering before falling back to BM25.
- **Hierarchical summaries** (cluster → summarize → corpus-level summary,
  no embeddings) for thematic/abstractive questions that neither keyword
  search nor fact extraction handles well on their own.
- **Incremental re-extraction pipeline**: detect changed docs and re-run
  extraction only for those, instead of any full-corpus rebuild.


The problem faced was because of **Gemini API Quota** Issue.
