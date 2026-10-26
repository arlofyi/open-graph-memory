# E7 frozen configuration (written before the TEST run, 2026-10-22)

DEV (conversations 0–1, 233 questions) was used for plumbing only. No
system's retrieval settings were tuned on it. Two adapter changes came out of
DEV, both in the competitors' favour, and both are frozen here:

1. **Graphiti gets a second read path.** Its basic `search()` (facts only)
   scored 73.8 at top 10. Graphiti's docs recommend `search_()`, and Zep feeds
   facts plus entity summaries, so `graphiti-combined` uses Graphiti's own
   `COMBINED_HYBRID_SEARCH_RRF` recipe over the same graph (81.1 on DEV).
   Both modes are reported.
2. **A top-200 cutoff is added (amendment to the pre-registration).** Cognee
   returns whole-session passages: at top 10 its answerer reads about 9,400
   tokens, against about 450–550 for every other system. Top 200 puts the
   other systems at a comparable budget, so the "matched context" gate is
   checked at top 200 for them and at top 10 for Cognee.

## Systems (TEST)

| name | system | read path |
|---|---|---|
| ogm-r2a | open-graph-memory @ this commit | rrf(dense, denseX, claims, facet), claims@2 |
| ogm-denseX | open-graph-memory | turn + claims embedding |
| ogm-dense | open-graph-memory | turn + previous turn embedding (baseline) |
| mem0 | mem0ai 2.2.1 (V3 pipeline), Qdrant | Mem0 default search |
| graphiti | graphiti-core 0.31.2, Neo4j 5.26 | `search()` |
| graphiti-combined | same graph | `search_(COMBINED_HYBRID_SEARCH_RRF)` |
| cognee | cognee 1.6.5 | `HYBRID_COMPLETION`, `only_context` |
| arlo-memory | arlo-memory main `d6a4922` via its Mem0-compatible server (chunks) | reference only; this path does not use its conversation-aware fact index |

Shared models: `claude-haiku-5-5` for all ingest-time LLM work, the answerer
and the judge; local `bge-m3` (1024-d) for every embedding. Harness: Mem0's
`memory-benchmarks` @ `4b61c5d`, unmodified, `--top-k 200 --top-k-cutoffs
10,50,200`, categories 1–4, conversations 2–9.

## DEV result (for the record; not a result)

| system | acc@10 | acc@50 | tokens@10 |
|---|---|---|---|
| ogm-r2a | 94.0 | 98.7 | 551 |
| ogm-denseX | 93.1 | 97.4 | 480 |
| ogm-dense | 85.8 | 94.8 | 433 |
| cognee | 94.8 | 95.7 | 9,421 |
| mem0 | 89.3 | 95.3 | 425 |
| arlo-memory | 88.4 | 95.3 | 432 |
| graphiti-combined | 81.1 | 89.7 | 816 |
| graphiti | 73.8 | 82.8 | 292 |

## Gates (unchanged)

- Pass: ogm-r2a beats the best public system (mem0, graphiti, graphiti-combined,
  cognee) at top 10 by ≥3 points with paired McNemar p < 0.05, and is not
  behind at matched context (ogm-r2a@200 vs cognee@10).
- Kill (as a public product): the best public system matches or beats ogm-r2a
  at top 10.
