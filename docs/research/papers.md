# Research: papers and repos relevant to open-graph-memory

Compiled 2026-10-19 from web sources. "Verified" means I read the number on the paper's own arXiv HTML or abstract page. **[UNVERIFIED]** means it came from a search snippet, a third-party summary, or a page I could not read in full.

Benchmark caveat for every LoCoMo number below. Most memory papers report end-to-end LLM-judge accuracy (J), not evidence recall. The memory-system rows also tend to be copied from the Mem0 paper's Table 1 rather than re-run. LoCoMo's answer key has known errors: one audit found about 6.4% of answers wrong **[UNVERIFIED, Penfield Labs blog]**. Category labels also differ between papers: A-MEM's F1 numbers appear under different category names in the A-MEM and Mem0 tables. Our harness should map category ids from the dataset file, not from any paper's prose.

---

## 1. HippoRAG and HippoRAG 2 (exact PPR setup)

**HippoRAG** — Gutiérrez et al., *HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs*, arXiv:2405.14831 (NeurIPS 2024). Verified from the HTML.
- An LLM runs OpenIE to extract triples. Noun phrases become nodes, and synonym edges are added when cosine similarity > **τ = 0.8**.
- Query NER entities are linked to nodes and used as PPR seeds. **Damping = 0.5.**
- Each seed's probability is multiplied by **node specificity s_i = 1/|P_i|**, where |P_i| is the number of passages the node came from. This is an IDF-like weight.
- Passages are scored by summing the PPR mass of their nodes. τ and the damping factor were tuned on only 100 MuSiQue training questions.
- R@5 vs ColBERTv2: MuSiQue 51.9 vs 49.2, 2Wiki 89.1 vs 68.2, HotpotQA 77.7 vs 79.3.
- Ablations (average R@5 72.9): without node specificity 70.9, without synonym edges 70.5.
- Using HippoRAG as IRCoT's retriever raised R@5 on 2Wiki from 74.4 to 93.9.

**HippoRAG 2** — Gutiérrez et al., *From RAG to Memory: Non-Parametric Continual Learning for LLMs*, arXiv:2502.14802 (ICML 2025). Verified from the HTML, Table 13 and Appendix G.
- **Graph:** phrase nodes plus **passage nodes**, with "context" edges from each passage to its phrases. Synonym threshold **0.8**. Implemented with python-igraph.
- **Linking is query-to-triple, not NER-to-node.** The whole query is embedded with NV-Embed-v2 and matched against triple embeddings, and the **top-5 triples** are kept.
- **Recognition memory:** an LLM (Llama-3.3-70B, prompt optimised with DSPy MIPROv2) filters those 5 triples to the relevant ones. If nothing survives, the system falls back to plain dense passage retrieval.
- **Reset vector:**
  - Each phrase in a surviving triple gets the average score of the triples it appears in. At most about 5 phrase seeds are kept. **[the exact cap rendered garbled in the HTML]**
  - **Every passage node is also a seed**, weighted by its dense query similarity × **0.05**.
  - Weight sweep (MuSiQue R@5): 0.01 → 79.9, **0.05 → 80.5**, 0.1 → 79.8, 0.3 → 78.4, 0.5 → 77.9.
- **Damping 0.5**, temperature 0. The QA reader gets the top-5 passages.
- Ablations (average R@5 87.1):
  - NER-to-node linking: 74.6
  - Query-to-node linking: 59.6
  - **Without passage nodes: 81.0**
  - Without the triple filter: 86.4 (the filter is the smallest contributor)

*For our PoC:* our claim↔entity bipartite graph already resembles HippoRAG 2 with "claim" playing the role of "triple". Five changes would bring it closer:
1. Link the **question to claims**, not to entities, using a dense index over claim text. That was worth +12.5 R@5 over NER linking.
2. Seed entities with the mean score of the claims they appear in.
3. **Add turn (passage) nodes, all seeded at about 0.05 × dense similarity.** Removing passage nodes cost about 6 points.
4. Use damping 0.5 and weight entity seeds by 1/|claims(entity)|.
5. Add synonym edges at a cosine threshold of 0.8. Our reversible entity resolution can treat these as soft "maybe-same" edges.

## 2. Graph and agent memory on LoCoMo and LongMemEval

**Zep / Graphiti** — Rasmussen et al., arXiv:2501.13956 (2025). Verified from the HTML.
- **Bitemporal edges** carry valid time (t_valid, t_invalid) and transaction time (t′created, t′expired).
- An LLM compares a new edge with semantically related existing edges. On a contradiction, it sets the old edge's t_invalid to the new edge's t_valid, so newer information always wins.
- **Episode nodes** keep raw messages, with bidirectional episode↔fact links for provenance.
- **Entity resolution:** a cosine and BM25 candidate search, then an LLM decides whether the entity is a duplicate. The decision is not reversible.
- **Search:** cosine + BM25 + n-hop BFS, combined with rerankers (RRF, MMR, episode-mention frequency, node distance, cross-encoder). No BFS depth is given.
- LongMemEval accuracy with gpt-4o: SS-pref 56.7, SS-asst 80.4, temporal 62.4, multi-session 57.9, knowledge-update 83.3, SS-user 92.9.
- LoCoMo J, from the Mem0 table: single-hop 61.7, multi-hop 41.4, open-domain 76.6, temporal 49.3.
- Zep's own blog argues *against* an append-only log of source ids, on the grounds that it "drifts from the current graph state". Their fix is a per-fact set of episode links that stays correct as the graph is mutated (getzep.com blog).

**Mem0 / Mem0ᵍ** — Chhikara et al., arXiv:2504.19413 (2025). Verified.
- **Graph pipeline (Mem0ᵍ):**
  - An entity extractor and a relation generator produce triples.
  - Embedding-threshold node matching handles entity resolution.
  - A conflict detector and an LLM update resolver **mark obsolete edges invalid** instead of deleting them.
- **Retrieval:** entity-centric subgraph expansion plus whole-query-vs-triplet similarity.
- **LoCoMo J (single / multi / open / temporal):**

| System | Single-hop | Multi-hop | Open-domain | Temporal |
|---|---|---|---|---|
| Mem0 | 67.1 | 51.2 | 72.9 | 55.5 |
| Mem0ᵍ | 65.7 | 47.2 | 75.7 | 58.1 |

- Overall J is 66.9 for Mem0 and 68.4 for Mem0ᵍ. Full context scores 72.9 overall.
- The graph helped temporal (+2.6) and open-domain, but hurt multi-hop (−4).

**A-MEM** — Xu et al., arXiv:2502.12110 (NeurIPS 2025).
- Zettelkasten-style notes, LLM-generated links, and "memory evolution" that rewrites linked notes.
- LoCoMo F1 with gpt-4o-mini, as reproduced in the Mem0 and Cognis tables: 27.0 / 12.1 / 44.7 / 45.9 under the Mem0 labels.
- The A-MEM paper itself prints these numbers under different category names. **[label mapping UNVERIFIED]**
- HyperMem's re-run gives J of 39.8 / 18.9 / 54.1 / 49.9. A-MEM is a weak baseline.

**MemoRAG** — Qian et al., arXiv:2409.05591.
- A light long-context "memory model" builds a global KV-compressed memory and drafts clues that guide retrieval. Trained with RLGF.
- The abstract names no LoCoMo or LongMemEval results. **[benchmarks UNVERIFIED]**
- Not relevant to a claim ledger, except that "draft clues, then retrieve" is a form of query expansion.

**HyperMem** — Yue et al., arXiv:2604.08256 (ACL 2026). Verified from the HTML.
- **Structure:** three levels (topic → episode → fact), held together by hyperedges. Topic hyperedges group episodes; fact hyperedges group the facts from one episode.
- **Retrieval:** every level has a BM25 index and a dense index (Qwen3-Embedding-4B). Each level fuses the two with **RRF** and reranks with Qwen3-Reranker-4B, running coarse to fine: 100 candidates in, then **10 topics → 10 episodes → 30 facts**.
- Hyperedge embedding propagation uses λ = 0.5. Answers come from GPT-4.1-mini.
- **LoCoMo J (single / multi / temporal / open / overall):**

| System | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
|---|---|---|---|---|---|
| HyperMem | 96.1 | 93.6 | 89.7 | 70.8 | 92.7 |
| HippoRAG 2 (re-run) | 86.4 | 75.9 | 78.5 | 66.7 | 81.6 |
| HyperGraphRAG | — | — | — | — | 86.5 |
| MemOS | — | — | — | — | 75.8 |

- No retrieval-recall metric is reported.
- *Takeaway:* hierarchical grouping (episode → its facts) is what fixes multi-hop completeness. Once you hit one fact, you pull its siblings.

**Cognis** — Daftari et al., arXiv:2604.19771 (2026). Verified.
- **No graph.** OpenSearch BM25 plus Matryoshka vectors: a 256-d shortlist of 200, rescored at 768-d.
- **Fusion:** RRF weighted 70% vector / 30% BM25, k = 10. Final score = **0.6 × fused + 0.4 × temporal score**, then a BGE cross-encoder rerank.
- Context-aware ingestion: before extraction, the top 10 similar memories are retrieved and the LLM chooses ADD / UPDATE / DELETE / NONE. Versions are linked with `is_current` and `replaces_id`.
- **LoCoMo J:** single-hop 72.0, multi-hop 51.0, open-domain 85.9, temporal 77.3.
- **LongMemEval accuracy (best answer model):**

| Type | Accuracy |
|---|---|
| SS-user | 100 |
| SS-assistant | 92.9 |
| SS-preference | 93.3 |
| Knowledge update | 96.2 |
| Temporal | 92.5 |
| Multi-session | 87.2 |
| **Overall** | **92.4** |

- **Retrieval:** Hit@K 0.83 overall. **Multi-session Hit@K is only 0.68** and temporal 0.80, the same weak spots we have.

**Entity-Collision** — Deng, arXiv:2605.29630 (May 2026). Abstract only.
- An evaluation protocol: distractors are built to share the answer's entity tokens, which pins a BM25 floor, and queries are stratified by tag.
- MiniLM-384 is the best embedder. BGE-large helps on intent-style queries but hurts on lexical ones.
- On LoCoMo, adaptive vector-weight routing gave **no gain despite 11.7 pp of oracle headroom**.
- The testbed is itself governed by an "event-sourced decision log" (prior art for item 5 below).
- *Actionable:* report recall in strata (lexical-overlap vs non-overlap questions) so a graph win is not just BM25 leakage.

**STITCH** — *Grounding Agent Memory in Contextual Intent*, arXiv:2601.10702 (Findings of ACL 2026).
- Indexes each step with a "contextual intent" cue (latent goal, action type, entity types) and filters retrieval by intent compatibility.
- Reports SOTA on CAME-Bench and LongMemEval, +35.6% over the best baseline. **[Per-category LoCoMo/LME numbers UNVERIFIED; the +35.6% may apply only to a CAME-Bench subset]**

**Other 2026 SOTA claims**
- **Agent Zero Memory** (arXiv:2608.29606): event timeline + entity-event graph + citation-locked fact store, with answers that may cite only evidence the reader opened. Claims **LongMemEval 95.6 / LoCoMo 93.6**, abstract only.
- **Hindsight** (arXiv:2512.12818): retrieval runs semantic, BM25, graph and temporal searches in parallel, then RRF and a cross-encoder. Claims LongMemEval 91.4 and LoCoMo 89.6. **[vendor-reported]**
- **EMem / EMem-G** (arXiv:2511.17208), verified:
  - Memory is LLM-extracted **elementary discourse units (EDUs) with turn-level source attribution**, linked to argument nodes, with synonym edges at cosine ≥ 0.9 and **PPR** over the graph.
  - Dense top-30 EDUs → LLM recall-filter → top-10 for QA.
  - LoCoMo J with gpt-4.1-mini: 0.853 overall (multi-hop 0.796, open-domain 0.717).
  - LongMemEval-S: 84.9.
  - **The LLM filter was the biggest single contributor** (−4.7 J without it).
  - The graph mainly helped temporal and knowledge-update questions; EMem without the graph is about 1–2 points lower.
  - This is the closest published analogue to our design.
- **Entity-Memory Graph Retrieval** (arXiv:2608.27925), abstract only:
  - The **only paper found that reports official LoCoMo evidence recall@k**.
  - Raw turns stay as Memory nodes, linked through shared Entity nodes plus chronological next-turn edges.
  - Retrieval: entity gating → semantic fusion → **one-hop chronological neighbour recovery** → dense backfill.
  - Recall@25 rose from 79.7 (dense) to 84.5 (graph), with gains from k = 5 to 50. Answer F1 showed no significant change.
  - 1,986 questions. This is our most direct baseline.

## 3. Proposition and claim-level indexing

- **Dense X Retrieval** (Chen et al., arXiv:2312.06648, EMNLP 2024), verified:
  - A proposition averages **11.2 words**, versus 58.5 for a 100-word passage. Propositions come from a Flan-T5-large "propositionizer" distilled from GPT-4.
  - R@5 gains: **+12.0 SimCSE, +9.7 Contriever (unsupervised); only +2.6 DPR / +2.8 GTR (supervised)**.
  - Gains are largest for rare entities (EntityQuestions DPR R@5 47.5 → 59.4).
  - *Implication:* with a strong modern embedder, expect a few points from finer granularity, not ten.
- **PropRAG** (Wang & Han, arXiv:2504.18070, EMNLP 2025), verified:
  - Replaces HippoRAG 2's triples with **propositions**. Each proposition joins its entities in a clique, an implicit hyperedge.
  - **Stage 1:** PPR with **damping 0.75** builds a 50-passage subgraph.
  - **Beam search over proposition paths:** B = 4, L_max = 3, M = 40 re-scored with a concatenated-path embedding. LLM-free at query time.
  - **Stage 2:** a second PPR with **damping 0.45**, seeded with the entities on the top paths. Passage weight λ = 0.05, synonym threshold 0.8.
  - R@5 vs HippoRAG 2: MuSiQue 78.3 vs 74.7, 2Wiki 94.1 vs 90.4, HotpotQA 97.4 vs 96.3.
  - *Directly applicable:* our claims *are* propositions with qualifiers.
- **LongMemEval** (Wu et al., arXiv:2410.10813, ICLR 2025), verified. This is the conversational evidence:
  - Indexing **facts alone as keys did not improve recall**. **Key = turn + extracted facts** did.
  - Round-level R@5 rose from 0.582 to 0.644 and R@10 from 0.692 to 0.784. The average gain was +9.4% recall and +5.4% QA.
  - Fact-level *values* hurt QA except on multi-session questions.
  - *For us:* score a turn by max(dense(turn ⊕ its claims)), and keep claims as graph keys while the **retrieved unit stays the turn**. That matches our turn-id evaluation.
- **"Same Ranking, Different Winner"** (arXiv:2605.24060), abstract only:
  - Systems store a source turn alongside derived facts. Changing which unit gets retrieval credit (Raw / Source / Canonical) changed nDCG on 83–94% of queries and flipped system rankings.
  - Source-linked credit was fully justified in only 29.2% of audited cases.
  - *For us:* our claim→turn mapping *is* a scoring-target choice. Document it, and audit a sample of credited turns.

## 4. Temporal retrieval and "as-of" retrieval

- **IA-RAG** (arXiv:2606.06044, June 2026):
  - Facts become Interval Event Units in a "Thematic Forest", ordered by **Allen's interval algebra**.
  - "Sub-graph Time Tightening" uses constraints in connected subgraphs to narrow fuzzy interval bounds. Retrieval traversal is guided by interval relations.
  - Evaluated on TimeQA, TempReason and ComplexTR (not conversational data). Reported 61.7 on TimeQA and +14.1 on compositional questions. **[numbers UNVERIFIED, third-party summary]** Code: xiaoAugenstern/LogicalRAG_TemporalQA.
  - *Takeaway:* our bitemporal qualifiers can support interval tightening, but this is a phase-2 idea.
- **"TimeRAG":** I found no paper with that exact name. The nearest matches:
  - **TG-RAG** (arXiv:2510.13590): a bi-level temporal graph that separates the same fact at different times.
  - **T-GRAG** (arXiv:2508.01680): a temporal KG plus temporal query decomposition.
  - **STAR-RAG** (arXiv:2510.16715): propagation over a time-aligned rule graph.
  - All three were seen at abstract level only.
- **LongMemEval time-aware query expansion**, verified: an LLM extracts a time range from the query and restricts retrieval to values whose event dates fall inside it. Temporal recall improved +11.3% for rounds, and R@10 rose from 0.550 to 0.722 with GPT-4o. It only works with a strong expansion LLM.
- **Graph-Native Bitemporal Memory Store** (arXiv:2607.26520), abstract only:
  - Immutable identity nodes link to versioned content nodes, each carrying valid and transaction intervals (Neo4j + HNSW).
  - On 60 LongMemEval questions: current-state search R@10 = 46.7 overall and 80 on knowledge-update.
  - The **time-travel filter dropped temporal-reasoning R@10 from 50 to 37.5** through "post-filter dilution".
  - *Lesson:* apply as-of filtering **before top-k truncation, or over-fetch**; never filter after cutting to k.
- **TSM** (arXiv:2601.07468): merges events into "durative memories" on a semantic timeline and retrieves by the query's temporal intent. Up to +12.2 absolute accuracy on LME and LoCoMo. **[mechanism details UNVERIFIED]**
- **Cognis's 0.6/0.4 fused + temporal score** and **Zep's "newer invalidates older"** are the cheap, proven fixes for currency.

*For currency:* when two claims share (subject, predicate) and differ in object, demote the superseded one at query time **unless the question has a past-tense or as-of cue**. Do not delete it; our ledger keeps the history.

## 5. Event-sourced and provenance-first KGs: is our novelty real?

What exists:
- **ActiveGraph**, *The Log is the Agent* (arXiv:2605.21997), verified: an append-only event log is the source of truth, and the graph is a deterministic projection whose replay gives identical state. It supports fork diffs between runs. It does **not** cover entity-resolution reversibility, bitemporal time, re-extraction diffs, or any QA benchmark.
- **KGK** (github.com/Yamai7354/kgk): an event-sourced, bitemporal (as_of / as_known_at) KG kernel with soft retraction and lineage. It has 3 commits, 0 stars, merges only (no split or undo), and no benchmarks.
- **projectmem** (arXiv:2606.12329): an append-only typed-event log projected into summaries, for coding agents.
- **Agent Zero Memory**: provenance and citation-locked facts with an event timeline. Its storage semantics are not in the abstract.
- **Zep/Graphiti:** bitemporal edges plus episode provenance, but a *mutable* graph with irreversible LLM merges. Zep argues against an append-only source-id log.
- **Mem0ᵍ:** invalidates edges instead of deleting them.
- **Entity-Collision's** testbed uses an event-sourced decision log.
- **Reversible ER prior art outside LLMs:**
  - Resolver trees with merge ids that can be undone (US patent 11960470).
  - Provenance-aware ER with coordinate-split-merge (Wang et al., DASFAA 2015).
  - Wikidata's claims + qualifiers + references model, from common knowledge (not fetched this session).
- **KG diffing between extractor versions:** nothing established. I found only graph-edit-distance comparisons of LM-extracted KGs (arXiv:2111.08546) and a pip package (drg-kg) that claims snapshot diffs.

**Verdict:** "append-only log + projected graph" and "bitemporal facts with provenance" are **not novel by themselves**. ActiveGraph, KGK and Zep each cover part of it. What I could **not** find anywhere is the full combination:
- (a) sourced, bitemporal claims with qualifiers as the ledger unit;
- (b) entity resolution as revisable, logged decisions with merge/split undo;
- (c) diffing projections between *extractor versions* over the same source turns;
- (d) an evaluation on LoCoMo or LongMemEval evidence recall.

Frame the novelty as that combination, especially (b)+(c), and cite ActiveGraph, KGK and Zep as partial prior art. This is a search-based negative, not proof of absence.

## 6. Cheap tricks for multi-hop recall

- **IRCoT** (Trivedi et al., arXiv:2212.10509, ACL 2023): alternate a CoT sentence with a retrieval that uses that sentence as the query. Reported up to +21 recall and +15 QA. With HippoRAG as the retriever, 2Wiki R@5 rose from 74.4 to 93.9. It costs one LLM call per hop; cap it at 2–3 hops.
- **Query→claim linking + LLM recognition filter** (HippoRAG 2, EMem): the filter was worth about 5 J in EMem.
- **Sibling expansion:** pull every claim from the same turn or session as a hit (HyperMem's fact hyperedges), plus **one-hop chronological neighbours** (2608.27925, +4.7 evidence R@25). This targets the all@k gap directly.
- **RRF fusion of graph + dense + BM25:** the default in Hindsight, HyperMem, Zep and Cognis. Cognis uses weighted RRF with k = 10. The standard RRF constant is k = 60 (Cormack et al., 2009; not re-fetched).
- **Two-stage PPR** (PropRAG): explore with damping 0.75, then exploit with 0.45 around the best paths.
- **Key expansion:** turn text ⊕ its claims as the dense key (LongMemEval).
- **Expansion depth:** the published sweet spot is shallow. Zep's BFS depth is unstated; PropRAG uses L_max = 3 (B = 3 was best on F1); 2608.27925 uses one hop. Deep BFS mainly adds distractors.

---

## Top 6 things to try in the PoC, ranked

1. **Port HippoRAG 2's reset vector exactly.** Dense query→claim top-5, average scores onto entities, multiply by 1/|claims(e)|, **seed every turn node at 0.05 × dense similarity**, damping 0.5. Turn nodes alone were worth about 6 R@5 in their ablation, and query-to-triple linking about 12.
2. **Add sibling and chronological expansion after PPR.** For each retrieved turn, add claims and turns from the same session within ±1 turn, then re-rank. This is the most direct lever on all@k completeness (+4.7 evidence R@25 on LoCoMo).
3. **Fuse with RRF:** PPR ranking + dense(turn ⊕ claims) + BM25 over turns, k = 60, tuning weights à la Cognis. Every 2026 SOTA system fuses; none relies on the graph alone.
4. **Currency demotion in the projection.** Mark claims superseded on the same (subject, predicate) as non-current. At query time, multiply them by about 0.3 unless the question carries a temporal or as-of cue. Apply the as-of filter *before* top-k, or over-fetch 3×.
5. **LLM recognition filter on the seeds:** a 20B model chooses which top-5 to top-10 linked claims are relevant before PPR, with a dense fallback when none survive. Cheap, and it was the largest single ablation in EMem.
6. **PropRAG-style beam over claim paths** (B = 4, L = 3) for multi-hop, or 2-hop IRCoT on hard questions only. Test against an oracle stratum so the gain is real (Entity-Collision protocol), and report the scoring target explicitly ("Same Ranking, Different Winner").

---

## Sources

- https://arxiv.org/html/2405.14831v3 (HippoRAG)
- https://arxiv.org/html/2502.14802v2 (HippoRAG 2)
- https://arxiv.org/html/2504.18070v2 (PropRAG)
- https://arxiv.org/html/2312.06648v3 (Dense X Retrieval)
- https://arxiv.org/html/2410.10813 (LongMemEval)
- https://arxiv.org/html/2501.13956v1 (Zep/Graphiti)
- https://www.getzep.com/blog/how-zep-tracks-provenance-in-agent-memory
- https://arxiv.org/html/2504.19413v1 (Mem0 / Mem0ᵍ)
- https://arxiv.org/abs/2502.12110 (A-MEM)
- https://arxiv.org/abs/2409.05591 (MemoRAG)
- https://arxiv.org/html/2604.08256v2 (HyperMem)
- https://arxiv.org/html/2604.19771v1 (Cognis)
- https://papers.cool/arxiv/2605.29630 and https://arxiv.org/pdf/2605.29630 (Entity-Collision)
- https://arxiv.org/pdf/2601.10702 (STITCH)
- https://arxiv.org/abs/2608.29606 (Agent Zero Memory)
- https://arxiv.org/abs/2512.12818 (Hindsight)
- https://arxiv.org/html/2511.17208v1 (EMem / EMem-G)
- https://arxiv.org/abs/2608.27925 (Entity-Memory Graph Retrieval, LoCoMo evidence recall)
- https://arxiv.org/abs/2605.24060 (Same Ranking, Different Winner)
- https://arxiv.org/abs/2606.24775 (Agent-native memory survey/benchmark)
- https://arxiv.org/pdf/2606.06044 and https://www.emergentmind.com/papers/2606.06044 (IA-RAG)
- https://arxiv.org/abs/2510.13590 (TG-RAG), https://arxiv.org/abs/2508.01680v1 (T-GRAG), https://arxiv.org/abs/2510.16715 (STAR-RAG)
- https://arxiv.org/abs/2607.26520 (Graph-native bitemporal memory store)
- https://arxiv.org/abs/2601.07468 (TSM)
- https://arxiv.org/html/2605.21997v1 (ActiveGraph, The Log is the Agent)
- https://github.com/Yamai7354/kgk (KGK)
- https://arxiv.org/html/2606.12329v1 (projectmem)
- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11960470 (resolver-tree merge/unmerge patent)
- https://openresearch-repository.anu.edu.au/entities/publication/a6809186-2daf-44d6-8041-2b73aa719870 (provenance-aware ER)
- https://arxiv.org/pdf/2111.08546 (KG extraction comparison via graph edit distance)
- https://arxiv.org/pdf/2212.10509 (IRCoT)
- https://penfieldlabs.substack.com/p/we-audited-locomo-64-of-the-answer (LoCoMo answer-key audit)
