# Landscape (October 2026)

| System | Core idea | Store of truth | Temporal | Where it's strong | What open-graph-memory does differently |
|---|---|---|---|---|---|
| **Graphiti / Zep** (Apache-2.0) | temporal KG for agents | the entity/edge graph | validity windows on edges, invalidation | time-aware agent memory | graph *is* the store: no replay, no undo of merges |
| **Cognee** | graph and vector built from many sources, MCP | the graph | limited | multi-source shared KG | same |
| **Mem0 (graph mode)** | facts plus an optional graph | memories | overwrite / update | simple adoption | no bitemporality |
| **HippoRAG 2** | PPR over a passage–entity graph | passages | none | **best multi-hop** (MuSiQue, WildGraphBench) | research code, not a product |
| **MS GraphRAG / LightRAG** | community summaries | summaries | none | global "what's the theme" questions | **score badly on temporal QA** |
| **HASH** | typed bitemporal KG + AI extraction | entity editions | full bitemporal | data model, provenance | claims feed a mutable entity store; here the claims are the only store (HASH's graph store is AGPL, its AI worker proprietary) |

Takeaways:

- Temporal plus provenance is already done well by Graphiti, so "a temporal
  knowledge graph" is not new on its own.
- The best retrieval idea (PPR over a bipartite graph) lives in research code. We
  apply it to **claims**.
- The unoccupied position is **event-sourced KG**: a ledger as truth, a projection
  as view, replay and diff, retractable merges.

Sources:
- [Cognee: best open-source AI memory tools 2026](https://www.cognee.ai/best-open-source-ai-memory-tools-for-llm-agents-and-developers) (vendor-authored)
- [Vectorize: best AI agent memory systems 2026](https://vectorize.io/articles/best-ai-agent-memory-systems)
- [Entity-Collision: attributing retrieval lift in agent memory (arXiv 2605.29630)](https://arxiv.org/pdf/2605.29630)
- [IA-RAG: interval-algebra temporal reasoning (arXiv 2606.06044)](https://arxiv.org/pdf/2606.06044)
- [RAGU: multi-step GraphRAG (arXiv 2607.11683)](https://arxiv.org/pdf/2607.11683)
- [HippoRAG 2 on WildGraphBench](https://benchmarklist.com/models/hipporag-2/)
- [ARIA: HASH, making AI legible (June 2026)](https://aria.org.uk/insights/hash-making-ai-legible-in-the-real-world)
