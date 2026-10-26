# Experiments

Every experiment here had its question, its metric and its **pass and kill
rules written down before it ran**. This page is that record. The results are
in [RESULTS.md](RESULTS.md).

Two rules applied throughout:

- **Tuning data and test data never mix.** All iteration happens on a tuning
  split (LoCoMo conversations 0–1). The test split (conversations 2–9, or the
  LongMemEval-S test sample) gets one run per frozen configuration.
- **Configurations are frozen in writing first.** Each one is written to a
  `results/*-FROZEN.md` file before its test run, and not changed after
  seeing test numbers.

## Status

| # | Question | Verdict |
|---|---|---|
| E0 | Can claims be extracted from chat at usable precision? | First extractor failed (68% strictly supported); second borderline pass (84–87%) |
| E1 | Does graph traversal over claims close the "all the evidence" gap? | Rule missed on test (multi-hop all@50 66.9 against 75), not killed (+23 over dense); overall all@10 75.8 against 61.8 for dense |
| E2 | Does resolving facts on valid time fix "stale version first"? | Pass (stale-first 2.6%, against 13.2% for a recency blend); top-1 mixed |
| E3 | Is entity linking conservative enough? | Passes (2.5 wrong per 1,000), but only 48% of links are right; the rule measured the wrong thing |
| E4 | Does the graph diff between two extractors explain itself? | Pass (98% explainable); caught a real extractor regression |
| E5 | Is rebuilding the graph fast enough? | Pass (0.31 s at 104k claims); the synonym pass needs a nearest-neighbour index |
| E6 (added) | Do recall gains become answer gains? | Yes: +10.1 judged accuracy at top 10 (60.6 to 70.7) |
| LME (added) | Does the recall gain hold on LongMemEval-S? | Same direction: turn all@10 88.6 to 95.5 (44 questions; directional only) |
| E7 | Head-to-head on LoCoMo | Kill fired as written: Cognee 90.3 against R2-A 89.6 at top 10 (p 0.50), with 16 times the context; at matched context R2-A 93.7 against 90.3 (p 7e-6); Mem0 and Graphiti 12–23 points behind |
| E8 | Head-to-head on LongMemEval-S | Pass: R2-A 83.3 at top 10, highest of eight; Mem0 80.0 is a tie (p 0.80); at matched context Cognee (73.3, p 0.013) and Graphiti combined (71.7, p 0.021) are behind; denseX 81.7 |

## E0–E5: the proof of concept

The baselines are arlo-memory's reported LoCoMo numbers (1,307 test
questions), plus our own dense baseline, which reproduces arlo-memory's dense
retrieval on the tuning split. E1 is decided on the tuning split first,
against a multi-hop all@50 baseline of 60.5, so that the test split is used
only once.

| # | Question | Metric | Baseline | Pass | Kill |
|---|---|---|---|---|---|
| **E0** | Can we extract claims from *conversation*? | claims per LoCoMo turn; precision on a judged sample of 100 | arlo-memory: 0 facts from 376 turns | at least 0.5 claims per turn at 85% precision or better | under 70% precision with gpt-oss-120b |
| **E1** | Does PageRank over claims close the completeness gap? | multi-hop **all@50** (decided on tuning data); overall **all@10** | tuning multi-hop all@50 60.5; test overall all@10 65.6 | multi-hop all@50 of 75 or more on tuning data, confirmed once on test | under +5 points on tuning data |
| E2 | Does bitemporal slicing fix "stale version first"? | how often an outdated version ranks first | 64.5% old-first; 15 of 38 stale versions still active | under 10% | no better than adding a recency term |
| E3 | Is entity resolution conservative enough? | wrong merges per 1,000 entities (judged) | n/a | under 5 | over 20 |
| E4 | Does the graph diff catch extractor regressions? | swap the extractor model; are the diffs explainable? | n/a | a reviewer can explain at least 90% of the top 50 changes | the diff is noise |
| E5 | How much does projection cost? | time to project 100k claims; incremental update p95 | n/a | under 30 s full, under 500 ms incremental | over 5 minutes full |

E2's fixture is a private set of supersession timelines from arlo-memory's
benchmark suite, so E2 cannot be reproduced from this repo alone.

## E7: head-to-head on LoCoMo

**Question.** Is open-graph-memory better than what anyone could install
today, on a benchmark someone else wrote?

**Harness.** Mem0's own LoCoMo runner (`mem0ai/memory-benchmarks` at
`4b61c5d`, Apache-2.0), run **unmodified**. Every system is reached the same
way, through the Mem0 OSS REST API that runner speaks (`POST /memories`,
`POST /search`). The runner decides how conversations are ingested (one
message per add, with the session timestamp), and it owns the answer and judge
prompts. We write only the adapters, so no system can win on our prompts.

**Systems.**

| system | version | how it is reached |
|---|---|---|
| Mem0 OSS | the runner's own server (`docker/mem0`), Qdrant | directly |
| Graphiti (the open-source engine behind Zep) | `graphiti-core`, latest on PyPI at run time | `bench/h2h/graphiti_server.py` (adapter) |
| Cognee | `cognee`, latest on PyPI at run time | `bench/h2h/cognee_server.py` (adapter), if it can run on the shared models; otherwise reported as not run, with the reason |
| arlo-memory | its main branch at run time | its own Mem0-compatible server |
| open-graph-memory: `dense`, `denseX`, `R2-A` | this repo | `bench/h2h/ogm-server.ts` (adapter) |

**Shared models.** Every system does its ingest-time LLM work with
`claude-haiku-5-5` and embeds with local `bge-m3` (1024 dimensions). Answerer
and judge are both `claude-haiku-5-5`. No system gets a bigger model than
another.

**Protocol.**
- The tuning conversations (0–1) are for plumbing and adapter bugs only. No
  system's retrieval settings are tuned on them; every system runs on its
  documented defaults.
- The configuration is frozen in `results/E7-FROZEN.md`, then the test
  conversations (2–9) run **once**.
- Scored categories are the runner's own (1–4). Cutoffs: top 10 and top 50
  results.
- Units differ: a Mem0 memory is a short fact, a Graphiti result is an edge,
  and an open-graph-memory result is one chat message. So the context the
  answerer reads is also reported in **tokens per question** at each cutoff,
  and systems are compared at matched context as well as at matched k.

**Rules.**
- **Pass:** at top 10, `R2-A` beats the best public system (Mem0, Graphiti or
  Cognee) on overall judged accuracy by at least 3 points, with a paired
  McNemar p < 0.05, **and** it is not behind at a matched token budget.
- **Kill (as a public product):** the best public system matches or beats
  `R2-A` at top 10.

**Amendments before the test run** (recorded in `results/E7-FROZEN.md`): a
top-200 cutoff was added so the matched-context comparison is possible, and
Graphiti got its recommended combined search as a second mode. The runner's
own Mem0 server was written against a `mem0ai` branch that no longer exists,
and against the released package it drops the session date, so Mem0 is
reached through `bench/h2h/mem0_server.py` instead, which gives it the date
the way the package supports (see that file).

## E8: head-to-head on LongMemEval-S

**Question.** Does E7's result hold on a benchmark that open-graph-memory was
**not** developed on, and that tests what LoCoMo barely does: facts that
change (knowledge updates), reasoning across many sessions, time, and
declining to answer when the memory does not exist?

**Why.** E7's caveat is home advantage: R2-A was developed on LoCoMo's tuning
conversations, the same dataset as E7's test. LongMemEval-S (Wu et al., ICLR
2025; `xiaowu0162/longmemeval-cleaned`) is longer (a median of 48 sessions,
about 490 messages and 120k tokens of history per question), its histories
are mostly off-topic filler, and Mem0 and Zep both publish numbers on it. The
only earlier contact was the retrieval-only check in RESULTS.md (45
questions, recall rather than judged answers).

**Harness.** The same pinned `mem0ai/memory-benchmarks` (`4b61c5d`), its
`benchmarks/longmemeval` runner, **unmodified**: it ingests one user and
assistant message pair per add with the session timestamp, and owns the answer
and judge prompts (including its per-type judging and abstention).

**Systems.** The eight E7 arms, at the same versions and configuration
(including an adapter fix that retries Cognee's graph build if it fails):
Mem0, Graphiti (basic and combined, one shared ingest), Cognee, arlo-memory,
and open-graph-memory `dense`, `denseX` and `R2-A`. Same shared models.

**Splits.**
- The 45 questions of the earlier recall check are already seen, so they
  never enter the test set.
- **Tuning set:** 6 of those 45 (the first 2 by question id of each of their
  three types: multi-session, temporal reasoning, knowledge update). For
  plumbing, adapter bugs and measuring ingest cost per system. No tuning.
- **Test set:** the runner's own stratified sampler (`--per-type K`, seed 42)
  over the remaining questions, all six types. The filtered dataset is a
  local, gitignored file passed with `--dataset-path`; the harness is not
  touched.
- **K** is the largest of 5, 10 or 20 that fits the budget at the per-question
  cost measured on the tuning set. It is written into `results/E8-FROZEN.md`
  before the test runs, and the test runs once.

**Metrics.** Judged accuracy at top 10, 50 and 200; context tokens per
question at each cutoff (characters / 4); per-type accuracy; abstention
accuracy; search latency; ingest spend per system per question.

**Matched context.** For each public system X: R2-A at the smallest cutoff
whose mean context is at least X's top-10 context, against X at top 10.

**Rules.** With at most 20 questions per type (120 in all), one question is
worth about 1 point, and only gaps of roughly 8 points or more will reach
p < 0.05. So the rules are on point estimates, paired McNemar p values are
reported alongside, and "significant" is claimed only at p < 0.05.
- **Pass (E7 holds off home turf):** R2-A has the highest top-10 accuracy of
  all eight arms, **and** is not behind any public system at matched context.
- **Kill (no "best" claim in public):** a public system beats R2-A by at least
  3 points at top 10 **and** at matched context.
- Anything in between is reported as mixed, with the per-type table, not
  rounded up.
- **Default read path (reported, no rule):** `denseX` (no LLM call per search)
  against the best public system at top 10. It decides whether `denseX` can be
  the default.

**Known limitation going in.** 95 of the 500 questions have two sessions with
the same timestamp. The open-graph-memory and Cognee adapters group a session
by timestamp, so those sessions merge into one. They share a date, so the
effect should be small.

**Amendments before the test run** (recorded in `results/E8-FROZEN.md`): a
fix to the Cognee adapter's result splitting, found on the tuning set, and
excluding whole question families (a question and its abstention rewrite) of
the seen questions, not just their ids.
