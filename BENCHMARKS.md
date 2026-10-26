# Benchmarks

How open-graph-memory is evaluated, and how to run every benchmark yourself.
The results are in [docs/RESULTS.md](docs/RESULTS.md); the rules each
experiment had to pass, written before it ran, are in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

There are two kinds of benchmark here:

- **Head-to-heads (E7, E8).** open-graph-memory against Mem0, Graphiti,
  Cognee and arlo-memory on **Mem0's own benchmark runner**, unmodified. The
  runner owns ingestion and the answer and judge prompts, so no system can win
  on our prompts.
- **Proof-of-concept experiments (E0–E6).** Our own scripts, measuring
  retrieval directly (does the evidence come back?) and the individual parts
  of the design.

## How the head-to-heads work

**The harness.** [`mem0ai/memory-benchmarks`](https://github.com/mem0ai/memory-benchmarks)
at commit `4b61c5d` (Apache-2.0), run unmodified: `bench/h2h/run.sh` refuses to
start if any tracked file in the clone differs from that commit. It talks to
each system through the Mem0 OSS REST API (`POST /memories`, `POST /search`),
so every system sits behind a small adapter in `bench/h2h/`:

| system | adapter | port (in `up.sh`) |
|---|---|---|
| Mem0 OSS 2.2.1 (Qdrant) | `mem0_server.py` | 8911 |
| Graphiti 0.31.2 (Neo4j), basic search | `graphiti_server.py` | 8912 |
| Graphiti, combined search (same graph) | `graphiti_server.py` | 8913 |
| Cognee 1.6.5 | `cognee_server.py` | 8914 |
| open-graph-memory `dense`, `denseX`, `R2-A` | `ogm-server.ts` | 8915, 8916, 8917 |

Each adapter's docstring says exactly what it does for its system. The two
that matter most: Mem0 gets the session date through its custom-instructions
hook (without it, every memory is dated "today"), and Graphiti gets its
recommended combined search as a second mode. arlo-memory is not public; it
was benchmarked through its own Mem0-compatible server, which is not in this
repo.

**Shared models.** Every system uses `claude-haiku-5-5` for every LLM call
(its own ingest-time extraction, plus the runner's answerer and judge), and
local `bge-m3` (1024 dimensions, through Ollama) for every embedding.

**What is measured.**
- **Judged accuracy at top k.** The runner gives the answerer each system's
  top k results (k = 10, 50, 200), and the judge marks the answer correct or
  not.
- **Context tokens at top k.** How much text the answerer read (characters /
  4). Systems return results of very different sizes, so this matters.
- **Matched context.** R2-A at the smallest cutoff whose context is at least
  the other system's top-10 context, against that system at top 10.
- **Significance.** A paired exact McNemar test on the questions where exactly
  one of the two systems was right. "Significant" means p < 0.05.
- **Ingest cost** per question, when each system's model calls go through a
  spend meter (see "Measuring cost" below).

## Running the head-to-heads

### 1. Install

You need git, [uv](https://docs.astral.sh/uv/), Docker, Node.js 24 or later,
and Ollama.

```bash
bench/h2h/setup.sh
```

This clones the harness at the pinned commit, makes one Python environment
per system under `.cache/h2h/venvs/` (versions pinned in
`bench/h2h/requirements-*.txt`), starts Qdrant and Neo4j in Docker on
localhost, and pulls `bge-m3`.

### 2. Get the data

Neither dataset is in this repo.

- **LoCoMo** (CC BY-NC 4.0): download `data/locomo10.json` from
  [snap-research/locomo](https://github.com/snap-research/locomo) to
  `.cache/locomo/locomo10.json`.
- **LongMemEval-S**: download `longmemeval_s_cleaned.json` from
  [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
  to `.cache/lme/longmemeval_s_cleaned.json`. The file we used has sha256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.

### 3. Start the servers

Point everything at an OpenAI-compatible endpoint that serves
`claude-haiku-5-5` (for example a [LiteLLM](https://github.com/BerriAI/litellm)
proxy in front of Anthropic's API):

```bash
export H2H_CHAT_BASE=http://127.0.0.1:4000/v1
export H2H_CHAT_KEY=...            # if your endpoint needs one
bench/h2h/up.sh
```

All seven servers listen on localhost only. Logs are in `.cache/h2h/logs/`.

### 4. LoCoMo (E7)

`bench/h2h/run.sh <name> <port> <dev|test>` runs the LoCoMo runner against one
system. `dev` is conversations 0–1, `test` is conversations 2–9 (1,307
questions).

```bash
bench/h2h/run.sh ogm-r2a           8917 test
bench/h2h/run.sh ogm-denseX        8916 test
bench/h2h/run.sh ogm-dense         8915 test
bench/h2h/run.sh mem0              8911 test
bench/h2h/run.sh graphiti          8912 test
bench/h2h/run.sh cognee            8914 test
H2H_RUN_ID=graphiti bench/h2h/run.sh graphiti-combined 8913 test   # searches graphiti's ingest
node bench/h2h/report.ts --split test                              # writes results/e7-h2h-test.json
```

`H2H_WORKERS` sets how many questions run in parallel (default 4). The
per-question results land in `.cache/h2h/test/`; they quote the dataset, so
keep them out of git. To browse them: `python3 bench/h2h/browse.py test`.

### 5. LongMemEval-S (E8)

```bash
python3 bench/h2h/e8-split.py      # builds .cache/lme/e8-test-pool.json (seen questions excluded)
A="--per-type 10 --seed 42"        # the runner's own stratified sampler: 60 questions
bench/h2h/run-lme.sh ogm-r2a  8917 .cache/lme/e8-test-pool.json test $A
bench/h2h/run-lme.sh mem0     8911 .cache/lme/e8-test-pool.json test $A
# ... and so on for every system, as in step 4.
bench/h2h/run-lme.sh graphiti 8912 .cache/lme/e8-test-pool.json test $A
mkdir -p .cache/h2h/e8-test/predicted_graphiti-combined
cp .cache/h2h/e8-test/predicted_graphiti/_ingestion_* .cache/h2h/e8-test/predicted_graphiti-combined/
H2H_RUN_ID=graphiti bench/h2h/run-lme.sh graphiti-combined 8913 .cache/lme/e8-test-pool.json test $A
python3 bench/h2h/report-lme.py test --json results/e8-lme-test.json
```

The 60 questions this draws are listed in
[`results/E8-FROZEN.md`](results/E8-FROZEN.md), so you can check you got the
same sample. The report refuses to score if any answer is empty: a failed
model call is recorded by the runner as an empty answer judged wrong, so
re-run those questions rather than scoring them.

### Measuring cost

To see what each system spends on ingest, give each one its own endpoint:
`up.sh` reads `H2H_CHAT_BASE_MEM0`, `H2H_CHAT_BASE_GRAPHITI`,
`H2H_CHAT_BASE_COGNEE`, `H2H_CHAT_BASE_DENSE`, `H2H_CHAT_BASE_DENSEX` and
`H2H_CHAT_BASE_R2A`, and the run scripts read `H2H_CHAT_BASE_QA` for the
answerer and judge. Each falls back to `H2H_CHAT_BASE`. A spend ledger in JSON
lines with `phase` (`e8-<split>-<system>`) and `usd` fields, passed as
`H2H_LEDGER`, adds ingest cost per question to the E8 report.

Rough costs at `claude-haiku-5-5` prices: LoCoMo about $67 for all systems;
LongMemEval-S from about $0.08 per question (open-graph-memory) to $1.70 per
question (Graphiti), for ingest alone.

## Running the proof-of-concept experiments

These need the LoCoMo file (above), Ollama with `bge-m3`, and
`OGM_CHAT_BASE` (plus `OGM_CHAT_KEY` if needed) pointing at an
OpenAI-compatible endpoint. Each script's header comment documents its
options. Every experiment reads a claim ledger that E0 writes to `.cache/ledger/<tag>/`,
so run E0 first.

```bash
# E0: extract claims from all 10 conversations and judge a sample of them.
# The benchmarked extractor is claims@2; its ledger tag in the scripts is haiku2.
OGM_EXTRACTOR=claims@2 node bench/e0-extract.ts --convs 0,1,2,3,4,5,6,7,8,9 --tag haiku2 --label all
# E1: evidence recall. Tune on conversations 0-1; run 2-9 once.
OGM_EXTRACTOR=claims@2 node bench/e1-recall.ts --tag haiku2 --convs 0,1 --label dev
# E6: judged answers from each read path's top 10 (reads the haiku2 ledger)
node bench/e6-qa.ts --k 10
# E3 and E5 read the untagged ledger: extract it with E0 without --tag first.
node bench/e3-resolution.ts --convs 0,1,2,3,4,5,6,7,8,9 --threshold 0.92
node bench/e5-cost.ts --target 100000
# E4 compares the untagged ledger with a second extraction by another model
node bench/e0-extract.ts --convs 0,1 --model claude-sonnet-5-5 --tag sonnet
node bench/e4-diff.ts --tag-b sonnet --convs 0,1
# LongMemEval-S retrieval check on the 45 seen questions (results/lme-seen-ids.json)
node bench/lme-recall.ts --phase extract && node bench/lme-recall.ts --phase eval
```

E2 (stale facts) runs on a private fixture from arlo-memory's benchmark
suite, so it cannot be reproduced from this repo.

## Reproducibility notes

- **The test splits are run once.** Each configuration is frozen in
  `results/*-FROZEN.md` before its test run.
- **LLM output is not fully deterministic.** Re-running will move individual
  answers, and with 60 questions a point or two either way is expected.
- **Model versions matter.** Every number here is for `claude-haiku-5-5` (and
  `claude-sonnet-5-5` as the E0–E6 judge).
- **Dataset licenses.** LoCoMo is CC BY-NC 4.0 (non-commercial). Check
  LongMemEval-S's license on its dataset page before using it.
- **Adapters are adapters.** Each system is reached through the runner's REST
  API, which may not be how you would integrate it natively.
