# E1 frozen configuration (written BEFORE the TEST run)

Frozen 2026-10-19, after DEV iteration v1 -> v2. TEST (convs 2-9) is
run exactly once with this config; no change after seeing TEST numbers.

- Primary method: `rrf(dense,claims,facet)`, RRF k=60, equal weights
  - dense: bge-m3 (Ollama, local), turn embedded with the previous turn
  - claims: claims@1 extracted by claude-haiku-5-5, ranked by cosine, emit source turns
  - facet: facets@1 claim tags + per-question plan (claude-haiku-5-5), typed traversal
- Secondary (reported, not decided on): dense, rrf(dense,claims), ppr(top20), rrf(dense,ppr)
- Synonym threshold 0.92, no turn adjacency.
- Gate (docs/EXPERIMENTS.md): multi-hop all@50 >= 75 pass; < +5 over baseline kill.
  Baselines: our dense (reproduces arlo-memory's dense baseline, DEV 53.5), arlo-memory's
  reported TEST all@10 65.6 overall (different embedder pipeline: comparable in unit, not identical).

DEV v2 (231 questions, multi-hop n=43): primary multi-hop all@50 76.7, overall all@50 90.9.

## Round 2 candidates (frozen 2026-10-19, before their TEST run)

After round 1 TEST (primary multi-hop all@50 66.5, gate not met), two new
candidates, chosen on DEV only (results/e1-dev-v3-c2.json), each run on TEST once:

- **R2-A** `rrf(dense,denseX,claims,facet)` on the **claims@2** ledger
  (denseX = turn + previous turn + the turn's claim texts, embedded once at ingest).
- **R2-B** `denseX` alone on claims@2: no query-time LLM call at all
  (the cheapest deployable form: claims only enrich the index).

DEV (claims@2): R2-A overall all@10/all@50 76.2/91.8, multi-hop 34.9/74.4;
R2-B 76.2/90.5, multi-hop 27.9/67.4.
