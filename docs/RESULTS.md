# Results

This page has every result we have measured, newest first:

1. **[LongMemEval-S head-to-head (E8)](#longmemeval-s-head-to-head-e8)**: open-graph-memory
   against Mem0, Graphiti, Cognee and arlo-memory on a benchmark it was not
   developed on.
2. **[LoCoMo head-to-head (E7)](#locomo-head-to-head-e7)**: the same systems on LoCoMo.
3. **[Proof-of-concept experiments (E0–E6)](#proof-of-concept-experiments-e0e6)**: does a
   claim ledger help retrieval at all, and which parts of the design earn their
   keep.

How each experiment was designed, and the pass and kill rules written down
before it ran, are in [EXPERIMENTS.md](EXPERIMENTS.md). How to run them
yourself is in [BENCHMARKS.md](../BENCHMARKS.md).

## The short version

- On **LongMemEval-S**, open-graph-memory's best read path (R2-A) has the
  highest top-10 accuracy of the eight systems tested: **83.3**, against 80.0
  for Mem0, 73.3 for Cognee and 71.7 for Graphiti. Mem0 is a statistical tie;
  Cognee and Graphiti are significantly behind when every system reads the same
  amount of context.
- On **LoCoMo**, R2-A ties Cognee at top 10 (89.6 against 90.3) while reading
  **1/16 of the context**, wins when both read the same amount, and is 12 to 23
  points ahead of Mem0 and Graphiti.
- `denseX`, the read path with **no LLM call per search** (8 ms), is within
  about 2 to 4 points of R2-A on both benchmarks and ahead of every other
  public system on LongMemEval-S.
- The idea that does the work is **claims as retrieval keys**, not graph
  traversal. Personalized PageRank over the graph added nothing (see the
  negative results below).

## Shared setup

- **Models.** Every system, ours and theirs, used `claude-haiku-5-5` for every
  LLM call (extraction, answering, judging), and local `bge-m3` (1024
  dimensions, through Ollama) for every embedding. In the proof-of-concept
  experiments the judges were `claude-sonnet-5-5`.
- **Data.** LoCoMo (CC BY-NC 4.0) and LongMemEval-S are read from the
  gitignored `.cache/` directory and are never committed. Neither are the
  per-question predictions, because they quote the datasets.
- **Raw numbers.** Every table here comes from a file in `results/`.

## LongMemEval-S head-to-head (E8)

**Setup.** Mem0's own benchmark runner (`mem0ai/memory-benchmarks` at commit
`4b61c5d`, `benchmarks/longmemeval`), run unmodified, so the answer and judge
prompts are Mem0's and not ours. 60 questions, 10 of each of the six question
types, including 3 abstention questions. They were drawn once by the runner's
own stratified sampler (seed 42), from the 452 questions we had never looked
at. Configuration: [`results/E8-FROZEN.md`](../results/E8-FROZEN.md). Raw
table: [`results/e8-lme-test.json`](../results/e8-lme-test.json).

Accuracy is the share of questions the judge marked correct, when the answerer
reads each system's top 10, 50 or 200 results. Tokens are the size of what the
answerer read (characters / 4).

| system | top 10 | top 50 | top 200 | tokens at top 10 | tokens at top 200 | ingest $ per question |
|---|---|---|---|---|---|---|
| **open-graph-memory R2-A** | **83.3** | 88.3 | 90.0 | 1,095 | 41,650 | 0.08 |
| open-graph-memory denseX | 81.7 | 88.3 | 86.7 | 2,281 | 50,627 | 0.08 |
| Mem0 OSS 2.2.1 | 80.0 | 80.0 | 88.3 | 550 | 11,749 | 0.56 |
| arlo-memory (Mem0-compatible API) | 75.0 | **91.7** | 93.3 | 3,697 | 68,566 | 0.06 |
| Cognee 1.6.5 | 73.3 | 75.0 | 75.0 | 16,802 | 17,922 | 0.25 |
| open-graph-memory dense | 73.3 | 83.3 | 90.0 | 2,140 | 50,177 | 0 |
| Graphiti 0.31.2, combined search | 71.7 | 90.0 | **95.0** | 2,010 | 43,691 | 1.70 (shared) |
| Graphiti 0.31.2, basic search | 36.7 | 40.0 | 38.3 | 383 | 7,970 | 1.70 (shared) |

Every system answered all 3 abstention questions correctly ("I don't know"
when the memory does not exist).

**By question type** (top 10; 10 questions each, so one question is 10 points):

| system | knowledge update | multi-session | single-session (assistant) | single-session (preference) | single-session (user) | temporal reasoning |
|---|---|---|---|---|---|---|
| R2-A | 90 | **70** | 80 | 90 | 80 | **90** |
| denseX | 90 | 50 | 90 | 100 | 80 | 80 |
| Mem0 | 100 | 60 | 80 | 80 | 90 | 70 |
| arlo-memory | 90 | 60 | 100 | 80 | 80 | 40 |
| Cognee | 90 | 50 | 100 | 90 | 70 | 40 |
| dense | 80 | 50 | 90 | 100 | 80 | 40 |
| Graphiti combined | 70 | 20 | 90 | 100 | 80 | 70 |
| Graphiti basic | 20 | 10 | 50 | 70 | 50 | 20 |

**Is the difference real?** We compare R2-A with each system question by
question (paired exact McNemar test): of the questions where exactly one of
the two was right, how often was it R2-A? At top 10:

| R2-A against | R2-A right, other wrong | other right, R2-A wrong | p |
|---|---|---|---|
| Mem0 | 9 | 7 | 0.80 |
| Cognee | 10 | 4 | 0.18 |
| Graphiti combined | 11 | 4 | 0.12 |
| Graphiti basic | 31 | 3 | 8e-7 |
| arlo-memory | 9 | 4 | 0.27 |
| denseX | 4 | 3 | 1.0 |
| dense | 9 | 3 | 0.15 |

**At matched context.** Systems return results of very different sizes: one
Cognee result is close to a whole conversation session, one Mem0 result is a
short fact. So we also compare at equal reading budgets: R2-A at the smallest
cutoff that gives the answerer at least as much text as the other system's
top 10.

| other system | its top 10 (tokens) | R2-A at (tokens) | R2-A | other | R2-A right, other wrong | other right, R2-A wrong | p |
|---|---|---|---|---|---|---|---|
| Mem0 | 550 | top 10 (1,095) | 83.3 | 80.0 | 9 | 7 | 0.80 |
| Cognee | 16,802 | top 200 (41,650) | 90.0 | 73.3 | 12 | 2 | 0.013 |
| Graphiti combined | 2,010 | top 50 (9,039) | 88.3 | 71.7 | 13 | 3 | 0.021 |
| Graphiti basic | 383 | top 10 (1,095) | 83.3 | 36.7 | 31 | 3 | 8e-7 |

**Against the rules written before the run:**
- **Pass: met.** R2-A has the highest top-10 accuracy of the eight systems,
  and it is not behind any public system at matched context. The rules were
  set on point estimates, because 60 questions cannot separate small gaps.
- **Kill: did not fire.** No public system beats R2-A at top 10.
- **What is significant.** At matched context, Cognee and Graphiti combined
  are significantly behind. **Mem0 is a statistical tie**: this run shows R2-A
  is not worse than Mem0, at about a seventh of Mem0's ingest cost, but not
  that it is better.
- **The cheap read path.** `denseX` scores 81.7 at top 10, ahead of every
  public system, with no LLM call per search. That makes it a reasonable
  default.

**What changed from LoCoMo.** Cognee, the strongest system on LoCoMo, drops to
73.3: its session-sized results fill the answerer's context with
LongMemEval's off-topic filler. Mem0 rises from 76.4 to 80.0. R2-A's lead
comes from temporal reasoning (90, against 40 to 70 for the others) and
multi-session questions (70, against 20 to 60), the two question types LoCoMo
barely tests.

**Caveats.**
- 60 questions is a small sample. One question is about 1.7 points, gaps under
  about 8 points are not significant, and each per-type cell is 10 questions.
- Search latency from this run is not comparable across systems:
  open-graph-memory and Cognee build their index on the first search, so the
  runner's timer includes ingest for them. Use the LoCoMo latencies below.
- Graphiti's two search modes share one ingest, so its cost is paid once for
  both.
- Ingest costs are from the spend meter and include retried work, so read
  them as upper bounds.
- Haiku is both answerer and judge.

## LoCoMo head-to-head (E7)

**Setup.** Mem0's own LoCoMo runner (`mem0ai/memory-benchmarks` at `4b61c5d`),
unmodified. 1,307 questions from conversations 2–9, run once.
Configuration: [`results/E7-FROZEN.md`](../results/E7-FROZEN.md). Raw table:
[`results/e7-h2h-test.json`](../results/e7-h2h-test.json).

| system | top 10 | top 50 | top 200 | tokens at top 10 | tokens at top 200 | search p50 |
|---|---|---|---|---|---|---|
| **open-graph-memory R2-A** | 89.6 | **93.3** | 93.7 | 512 | 8,565 | 1,297 ms |
| Cognee 1.6.5 | **90.3** | 91.9 | 92.3 | 8,335 | 9,846 | 198 ms |
| open-graph-memory denseX | 86.2 | 93.0 | **94.0** | 433 | 8,000 | 8 ms |
| arlo-memory (Mem0-compatible API) | 84.5 | 91.1 | 93.5 | 412 | 7,217 | 92 ms |
| open-graph-memory dense | 80.3 | 90.3 | 93.4 | 389 | 7,740 | 7 ms |
| Graphiti 0.31.2, combined search | 77.3 | 87.0 | 89.9 | 722 | 9,118 | 432 ms |
| Mem0 OSS 2.2.1 | 76.4 | 89.3 | 92.7 | 433 | 8,712 | 118 ms |
| Graphiti 0.31.2, basic search | 66.4 | 72.9 | 75.9 | 292 | 6,460 | 337 ms |

Latency was measured with every system running at once on one laptop, and the
open-graph-memory server keeps everything in memory, so read it as relative.
R2-A's 1.3 seconds is almost all one LLM call that plans the query.

**Is the difference real?** R2-A against each system at top 10 (R2-A right and
the other wrong / the other right and R2-A wrong): Cognee 67/76 (p 0.50),
arlo-memory 99/32 (4e-9), Graphiti combined 210/49 (6e-25), Mem0 228/56
(9e-26), Graphiti basic 346/43 (6e-60).

**At matched context.** R2-A at top 200 (8,565 tokens) against Cognee at top
10 (8,335 tokens): **93.7 against 90.3**, 72/27, p = 7e-6.

**Against the rules written before the run:**
- **Pass: not met.** The pass rule asked R2-A to beat the best public system
  by at least 3 points at top 10. It trails Cognee by 0.7, which is not
  significant.
- **Kill: fired as written.** The kill rule was "the best public system
  matches or beats R2-A at top 10", and Cognee does. It does so by returning
  whole sessions: Cognee sizes its chunks from the embedder's limit (8,192
  tokens for bge-m3), so its ten results are 16 times R2-A's context.
- **Read with matched context:** R2-A beats Cognee at equal context,
  significantly, and beats every other public system at top 10 by 12 to 23
  points. So the honest claim from LoCoMo is not "best at top 10". It is **as
  accurate as the strongest public system at 1/16 of the context, and more
  accurate at equal context.** That is why we ran LongMemEval-S next.

**Caveats.**
- Haiku is both answerer and judge.
- Every public system ran on its documented defaults. The adapters give Mem0
  the session date (through its custom-instructions hook) and add Graphiti's
  combined search as a second mode. Graphiti's default cross-encoder reranker
  could not run, because the chat endpoint we used returns no logprobs.
- open-graph-memory and Cognee index a session at a time; the others index
  each message.
- R2-A was developed on LoCoMo conversations 0–1, so LoCoMo is its home turf.
  The test conversations (2–9) were never used for tuning.
- arlo-memory was reached through its Mem0-compatible API, which does not use
  its conversation-aware fact index.

## Proof-of-concept experiments (E0–E6)

These came first. They ask whether a claim ledger helps retrieval at all, and
they score **retrieval** directly: for each LoCoMo question, do *all* of its
evidence turns come back in the top k? Everything was tuned on conversations
0–1 only; each configuration was frozen in
[`results/E1-FROZEN.md`](../results/E1-FROZEN.md) before its single run on
conversations 2–9. Precision of extracted claims was judged by an LLM, not a
human, so read it as an estimate: with 150 judged claims the margin is about ±6
points.

### Evidence recall on LoCoMo (conversations 2–9)

`any@10` means at least one evidence turn is in the top 10; `all@10` means all
of them are.

| method | any@10 | any@50 | all@10 | all@50 | multi-hop all@50 |
|---|---|---|---|---|---|
| dense (bge-m3, each turn with the previous one) | 75.6 | 92.0 | 61.8 | 80.2 | 43.5 |
| arlo-memory, its own reported number | 79.5 | 93.7 | 65.6 | 82.8 | n/a |
| **R1** `rrf(dense, claims, facet)`, first extractor | 87.4 | 96.9 | 73.5 | 89.7 | 66.5 |
| **R2-A** `rrf(dense, denseX, claims, facet)` | **88.7** | **97.1** | **75.8** | 89.4 | **66.9** |
| **R2-B** `denseX` only (no LLM call at search time) | 84.8 | 96.1 | 72.2 | 87.4 | 61.5 |

The arlo-memory row comes from a different pipeline (a hosted bge-m3 and its
own chunking). It is comparable in unit (one turn) and split, but it is not the
same system; the dense row is the controlled baseline.

### Each experiment against its rule

| # | Rule set before the run | Result | Verdict |
|---|---|---|---|
| E0 | at least 0.5 claims per turn, at least 85% supported by their sources | First extractor: 0.89 per turn, only **68%** strictly supported. Second extractor (one added provenance rule): **84.0%** across all 10 conversations, at 0.87 per turn | First extractor **failed**; second **borderline pass** (within the margin) |
| E1 | multi-hop all@50 at least 75, on tuning data and then on test | Tuning 76.7, test **66.5** (R1) and **66.9** (R2-A); dense 43.5 | **Not passed, not killed.** +23 points over dense, but below 75 on test |
| E2 | the stale version of a fact ranked first under 10% of the time | dense 60.5%, dense + as-of filter + recency 13.2%, **open-graph-memory 2.6%** (0% on the held-out slice). Top-1: recency 84.2 vs open-graph-memory 82.9 | **Pass** on the rule; mixed on top-1 |
| E3 | under 5 wrong entity links per 1,000 entities | 2.5 per 1,000, **but only 48% of links are right**. Merging them would have created clusters of up to 7 | Passes **only because linking is sparse.** The rule measured the wrong thing; "link, never merge" is the right call |
| E4 | at least 90% of the graph diff between two extractors is explainable | **98%**. Swapping Haiku for Sonnet as the extractor drops claims that were 88% real | **Pass**: the diff caught a real regression |
| E5 | a full rebuild under 30 s at about 100k claims | **0.31 s** for 104k claims; 32 ms per graph query; 472 MB of memory. The synonym pass, which compares every pair of entities, is the bottleneck | **Pass**; synonyms need an approximate nearest-neighbour index |

### What the experiments taught us

1. **Claims help retrieval a lot; graph traversal does not, yet.**
   - `rrf(dense, claims)` alone gives most of the lift.
   - Personalized PageRank (as in HippoRAG 2) is *worse* than dense on its own
     (all@10 58.4), and fused with dense it is no better than fusing claims.
     On LoCoMo nearly every claim hangs off one of two speakers, and those hub
     entities swamp the spreading.
2. **The graph that does work is typed traversal.**
   - LoCoMo's "multi-hop" questions are mostly *aggregation* ("What activities
     does Melanie do?"), with the evidence spread over many sessions.
   - A coarse view of 16 facets, plus one LLM call that plans the query and
     walks `(entity)-[facet]→claims`, moved multi-hop from 72 to 77 on the
     tuning data. On test it held up only as +2.
3. **The cheapest win is enriching the index.** Embedding each turn together
   with its claims (`denseX`) needs no LLM call at search time and no graph,
   and it beats the dense baseline's all@10 by 10 points.
4. **Multi-hop plateaus at about 67 (all@50), and the cause is ranking, not
   extraction.**
   - The claims cite *all* the evidence turns for about 90% of multi-hop
     questions, yet only about 67% retrieve all of it in the top 50.
   - So about 23 points of the gap is aggregation the retrieval still misses.
     The next lever is better aggregation (a better entity anchor, or an
     "everything about X in facet F" call), not more extraction.
5. **Resolving facts on valid time fixes staleness, not top-1.** It all but
   eliminates "old version above new", but loses some top-1 to distractors on
   other topics, because slot heads are ranked by similarity only.
6. **Provenance is a prompt problem, and solvable.** The first extractor's
   failures were true facts citing too few turns. One explicit rule took strict
   precision from 72% to 87% on the tuning data.
7. **Replay and diff earns its keep.** The "bigger model" extractor upgrade
   would have silently dropped hundreds of true claims; the diff showed it, and
   the retrieval numbers showed which read paths cared. Entity names churn a
   lot between extractors (only 97 of about 450 shared), which argues for a
   stable mention-to-entity layer.

### Tried and not kept

| idea | where | result |
|---|---|---|
| HippoRAG 2 PageRank (damping 0.5, top-5 query-to-claim seeds, passage weight 0.05) | E1 | worse than dense alone (test all@10 58.4 vs 61.8); fused with dense, no better than fusing claims |
| PageRank with entity specificity, wider seeds (top 20), passage weight 0.3 | E1 tuning | ±1 point; the two speakers dominate |
| Next-turn expansion (`dense+next`, a cheap baseline from the literature) | E1 | **hurts** at every k (test all@50 77.1 vs 80.2): it spends rank slots on filler |
| Synonym threshold 0.8 (HippoRAG 2's default) | E1 | bge-m3 similarities run high, so 0.8 linked almost every entity to every other (66k links for 400 entities). 0.92 is needed |
| Diversity (MMR) inside the facet leg for aggregation questions | E1 tuning | no change (74.4 either way) |
| Grouping fact versions by subject plus similarity | E2 tuning | revisions phrase subjects differently, so every group had one member (46% stale-first). Grouping by similarity alone fixed it |
| Blending recency or document context into slot heads | E2 tuning | no gain or worse |

### Do recall gains become answer gains? (E6)

The top 10 turns from each method go to one answer call and one judge call,
both `claude-haiku-5-5`, with this repo's own prompts (so the absolute numbers
are not comparable with the head-to-heads above). 1,307 test questions.

| method (top 10) | all | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|---|
| dense | 60.6 | 36.4 | 57.8 | 30.1 | 73.0 |
| **R2-A** | **70.7** | **54.4** | **72.1** | **34.9** | **79.6** |
| R2-B (`denseX`) | 68.4 | 47.3 | 69.4 | 30.1 | 79.4 |

Yes: **+10.1 accuracy at the same 10-turn context**, and +18 on multi-hop.

### LongMemEval-S recall check

A first, retrieval-only look at LongMemEval-S, before E8: 45 questions picked
with a fixed seed (15 each of multi-session, temporal reasoning and knowledge
update), 2,064 history sessions. Session-level recall is saturated (every
method finds the right session in the top 10, because the filler sessions are
off-topic), so we score turns instead: the user turns the dataset marks as
holding the answer (44 questions).

| method | all@10 | all@20 | multi-session all@10 | temporal all@10 | knowledge update all@10 |
|---|---|---|---|---|---|
| dense | 88.6 | 95.5 | 93.3 | 86.7 | 85.7 |
| denseX | 93.2 | 95.5 | 93.3 | 93.3 | 92.9 |
| rrf(dense, claims) | 93.2 | 97.7 | 93.3 | 93.3 | 92.9 |
| **R2-A** | **95.5** | **100** | 93.3 | **100** | 92.9 |

The direction matches LoCoMo (+6.9 all@10), but one question is worth about
2.3 points here. These 45 questions were excluded from E8.

## Cost

All costs are for `claude-haiku-5-5` (and `claude-sonnet-5-5` as the judge in
E0–E6). Embeddings ran locally and cost nothing.

- **Proof of concept (E0–E6):** $7.15 for about 13,000 model calls, almost
  all of it claim extraction, query planning, answering and judging.
- **LoCoMo head-to-head (E7):** $67 for all eight systems.
- **LongMemEval-S head-to-head (E8):** ingest cost per question is in the E8
  table above. Graphiti is the most expensive system to ingest into, because it
  makes several LLM calls for every message it stores.
