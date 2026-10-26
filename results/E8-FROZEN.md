# E8 frozen configuration (2026-10-23, before TEST)

The pre-registration is in `docs/EXPERIMENTS.md` (E8). This file freezes what
TEST runs, after DEV, and records what DEV changed. TEST runs once.

## TEST

- **Sample:** the harness's own stratified sampler,
  `sample_questions_stratified(per_type=10, seed=42)`, over
  `.cache/lme/e8-test-pool.json`. That gives **60 questions, 10 of each of the
  six types**, 3 of them abstention. The IDs are listed below, and the run is
  invoked as `--dataset-path .cache/lme/e8-test-pool.json --per-type 10`.
- **Data:** `longmemeval_s_cleaned.json`, sha256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`. Local
  and gitignored, never committed.
- **Harness:** `mem0ai/memory-benchmarks` at `4b61c5d`,
  `benchmarks/longmemeval/run.py`, unmodified. It is driven through
  `bench/h2h/run-lme.sh` with cutoffs 10, 50, 200 and Haiku 5.5 as answerer and
  judge.
- **Systems:** the eight E7 arms at the E7 versions. The seven public arms are
  started by `bench/h2h/up.sh`; arlo-memory runs its own Mem0-compatible server:

  | arm | port |
  |---|---|
  | Mem0 | 8911 |
  | Graphiti basic | 8912 |
  | Graphiti combined (searches Graphiti basic's ingest) | 8913 |
  | Cognee | 8914 |
  | OGM `dense` | 8915 |
  | OGM `denseX` | 8916 |
  | OGM `R2-A` | 8917 |
  | arlo-memory | 8918 |

  TEST reuses the DEV servers and stores. Every runner user ID is per question
  and per system, so DEV and TEST data never mix.
- **Empty answers are never scored.** If a model call fails, the runner records
  an empty answer judged FAIL. Every result file is checked for empty answers
  before scoring; any such question is answered again, never scored.

## What DEV changed (amendments, all before TEST)

1. **Cognee adapter bug, fixed.** `_split` broke Cognee's context on blank
   lines and on any `## ` line. LongMemEval's assistant replies hold both, so
   one Cognee passage became many fragments. DEV top-10 passages had a median of
   234 characters, and Cognee answered 1 of 6 questions. The adapter now splits
   only on Cognee's own section headers and its `---` passage separator. After
   the fix: a median of 10,096 characters, and 6 of 6.
   - **E7 is not materially affected.** Only 12 of LoCoMo's 5,882 turns contain
     a blank line, and none contains `## ` or `---`. E7's Cognee top-10
     passages had a median of 3,087 characters, with 47 of 13,070 under 200.
2. **Question families excluded, not just question IDs.** An `_abs` question
   is a rewrite of an original question on the same history. Excluding the 45
   seen IDs exactly let two siblings of seen questions into the sample
   (`031748ae`, `09ba9854_abs`). The pool now excludes the seen questions'
   whole families: 452 questions, not 455.
3. **Where the seen IDs live.** They are in `results/lme-seen-ids.json`;
   `results/lme-subset45.json` holds only that check's summary.

## DEV (6 questions; plumbing and cost only)

| arm | top-10 correct (after the Cognee fix) | ingest $ per question | context tokens at top 10 / 200 |
|---|---|---|---|
| OGM `R2-A` | 6/6 | 0.10 | 1,002 / 41,385 |
| Cognee | 6/6 | 0.24 | 21,328 / 22,414 |
| OGM `denseX` | 5/6 | 0.08 | 2,375 / 51,111 |
| OGM `dense` | 5/6 | 0 | 2,358 / 49,818 |
| Mem0 | 5/6 | 0.50 | 597 / 11,260 |
| arlo-memory | 5/6 | 0.06 | 3,995 / 72,145 |
| Graphiti basic, combined | 1/1, 1/1 (the cost probe question only) | 1.20 | 380 / 8,314 (basic) |

Six questions decide nothing. This table exists to show
the plumbing works and to price TEST.

## Gates (unchanged from the pre-registration)

- **Pass:** `R2-A` has the highest top-10 accuracy of the eight arms, **and**
  is not behind any public system at matched context.
- **Kill:** a public system beats `R2-A` by 3 points or more at top 10
  **and** at matched context.
- **Mixed:** anything else.
- **Matched context:** `R2-A` at the smallest cutoff whose mean context is at
  least the rival's top-10 context, against the rival at top 10.
- **Statistics:** paired exact McNemar p values are reported, and a result is
  called significant only when p < 0.05.

## TEST question IDs

`06878be2`, `06db6396`, `099778bb`, `0a995998`, `0ddfec37_abs`, `0e4e4c46`, `0edc2aef`, `0f05491a`, `1568498a`, `19b5f2b3`, `1a1907b4`, `1c0ddc50`, `1da05512`, `21d02d0d`, `22d2cb42`, `25e5aa4f`, `2698e78f`, `29f2956b_abs`, `2a1811e2`, `38146c39`, `3ba21379`, `3e321797`, `3f1e9474`, `4388e9dd`, `5025383b`, `505af2f5`, `51a45a95`, `5d3d2817`, `75499fd8`, `75832dbd`, `8752c811`, `8aef76bc`, `8b9d4367`, `8c18457d`, `8cf4d046`, `8ebdbe50`, `8fb83627`, `a89d7624`, `aae3761f`, `ac031881`, `ad7109d1`, `b6019101`, `bf659f65`, `c14c00dd`, `c7cf7dfd`, `cc6d1ec1`, `ceb54acb`, `d596882b`, `dd2973ad`, `eace081b`, `eeda8a6d_abs`, `fca70973`, `gpt4_1e4a8aeb`, `gpt4_2487a7cb`, `gpt4_4cd9eba1`, `gpt4_6ed717ea`, `gpt4_7ca326fa`, `gpt4_af6db32f`, `gpt4_d6585ce9`, `gpt4_e05b82a6`
