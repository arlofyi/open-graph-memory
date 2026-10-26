# open-graph-memory

**Memory for AI agents, built on a ledger of sourced facts instead of a
mutable graph.**

Most agent-memory systems turn conversations into a knowledge graph and then
edit that graph in place. When the extractor gets something wrong, merges two
different people into one, or a better model comes along, the mistake is baked
in. open-graph-memory keeps an **append-only ledger of claims** (small,
sourced, time-stamped facts) as the only source of truth. The entity graph is
a view computed from the ledger, so it can be thrown away and rebuilt at any
time.

This repository is a research proof of concept: the ideas, the experiments
that test them, and the benchmark harness, with every result reproducible.

## How it compares

We ran open-graph-memory against Mem0, Graphiti (the engine behind Zep),
Cognee and arlo-memory on **Mem0's own benchmark harness, unmodified**, with the
same models for every system (`claude-haiku-5-5` for every LLM call, local
`bge-m3` embeddings). Accuracy is the share of questions answered correctly
from each system's top 10 results.

| system | LongMemEval-S | LoCoMo | context read at top 10 (tokens, LoCoMo) |
|---|---|---|---|
| **open-graph-memory R2-A** | **83.3** | 89.6 | 512 |
| open-graph-memory denseX | 81.7 | 86.2 | 433 |
| Mem0 OSS 2.2.1 | 80.0 | 76.4 | 433 |
| arlo-memory | 75.0 | 84.5 | 412 |
| Cognee 1.6.5 | 73.3 | **90.3** | 8,335 |
| Graphiti 0.31.2 (combined search) | 71.7 | 77.3 | 722 |
| Graphiti 0.31.2 (basic search) | 36.7 | 66.4 | 292 |

What that means, honestly:

- **LongMemEval-S** (60 questions; open-graph-memory was not developed on
  it): R2-A has the highest accuracy. Mem0 is a statistical tie. Cognee and
  Graphiti are significantly behind when every system reads the same amount of
  context.
- **LoCoMo** (1,307 questions; R2-A was developed on other conversations from
  this dataset): Cognee ties R2-A, but reads 16 times as much context to do
  it. Given the same reading budget, R2-A wins clearly (93.7 against 90.3,
  p = 7e-6). Mem0 and Graphiti are 12 to 23 points behind.
- **denseX**, the read path with no LLM call per search (8 ms), is within a
  few points of R2-A on both and ahead of every other public system on
  LongMemEval-S.

Full tables, per-type breakdowns, significance tests and caveats are in
[docs/RESULTS.md](docs/RESULTS.md). The rules each experiment had to pass were
written down before it ran: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## The idea in four rules

1. **Append-only.** Nothing in the ledger is edited or deleted. A correction is
   a new claim that retracts or supersedes an old one.
2. **Every claim is sourced.** A claim cites the messages or document spans it
   came from. No source, no claim.
3. **Every claim has two times.** When it was true in the world (valid time),
   and when we learned it (recorded time). "Where does Ana live *now*?" and
   "what did we believe last March?" are different questions.
4. **Everything derived is reproducible.** The graph, the entity profiles and
   the decisions to link two mentions are all computed from the ledger. Same
   ledger, same graph.

What that buys you:

- **Upgrade the extractor without migrating.** Re-extract with a better model,
  and get a *diff* of what changed in the graph before you accept it. In our
  tests, a "bigger model" upgrade would have silently dropped hundreds of true
  facts; the diff caught it.
- **Undo a bad merge.** Two mentions are linked by a `same_as` claim, which can
  be retracted like any other. Mentions are linked, never merged.
- **Keep contradictions.** Conflicting claims stay in the ledger with their
  sources; nothing is silently overwritten.
- **Stale facts stop winning.** Resolving facts on valid time cut "the outdated
  version ranked first" from 60.5% (dense retrieval) to 2.6%.

```
messages, documents
        │  extract (one LLM call per session)
        ▼
  CLAIM LEDGER  ── append-only, sourced, bitemporal
        │
        ├── project(ledger, lens) ──▶ entity graph (a disposable view)
        ├── replay(new extractor) ──▶ graph diff
        └── retrieve(question)    ──▶ the turns and claims that answer it
```

The full design is in [docs/CONCEPT.md](docs/CONCEPT.md).

## Read paths

Retrieval returns conversation turns, ranked three ways:

| read path | how it ranks | LLM call per search | latency | LoCoMo top 10 |
|---|---|---|---|---|
| `dense` | each turn embedded with the previous one | no | ~7 ms | 80.3 |
| `denseX` | each turn embedded together with the claims extracted from it | no | ~8 ms | 86.2 |
| `R2-A` | reciprocal rank fusion of dense, denseX, a claim search and a typed facet walk planned by one LLM call | yes | ~1.3 s | 89.6 |

`denseX` is the sensible default. R2-A is for when accuracy matters more than
a second of latency.

## Quickstart

You need **Node.js 24 or later** (it runs the TypeScript directly, with no
build step and no dependencies), **Ollama** with the `bge-m3` model, and an
**OpenAI-compatible chat endpoint** for claim extraction (for example a
[LiteLLM](https://github.com/BerriAI/litellm) proxy in front of Anthropic's
API).

```bash
git clone https://github.com/arlofyi/open-graph-memory.git
cd open-graph-memory
npm test                       # unit tests; no network needed
ollama pull bge-m3
```

Run it as a memory server. It speaks the Mem0 OSS REST API, so anything that
talks to Mem0 can talk to it:

```bash
export OGM_CHAT_BASE=http://127.0.0.1:4000/v1   # your OpenAI-compatible endpoint
export OGM_CHAT_KEY=...                          # if the endpoint needs a key
OGM_H2H_MODE=denseX OGM_EXTRACTOR=claims@2 node bench/h2h/ogm-server.ts --port 8917
```

```bash
curl -s -X POST localhost:8917/memories -H 'content-type: application/json' -d '{
  "user_id": "ana", "timestamp": 1700000000,
  "messages": [{"role": "user", "content": "I moved to Lisbon last spring and adopted a cat named Miso."}]}'

curl -s -X POST localhost:8917/search -H 'content-type: application/json' \
  -d '{"user_id": "ana", "query": "What is my cat called?", "limit": 3}'
```

Messages with the same `timestamp` form one session. Claims are extracted a
session at a time, on the first search after new messages arrive. The server
keeps everything in memory, has no authentication, and only binds to
localhost.

### Configuration

| variable | default | what it sets |
|---|---|---|
| `OGM_CHAT_BASE` | none (required for extraction) | OpenAI-compatible chat-completions endpoint |
| `OGM_CHAT_KEY` | `local` | bearer token for that endpoint |
| `OGM_MODEL` | `claude-haiku-5-5` | model used by the server for extraction and planning |
| `OGM_EXTRACTOR` | `claims@1` | extractor prompt version; `claims@2` is the one benchmarked |
| `OGM_H2H_MODE` | `r2a` | server read path: `dense`, `denseX` or `r2a` |
| `OGM_EMBED_BASE` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OGM_EMBED_MODEL` | `bge-m3` | embedding model |
| `OGM_CACHE` | `.cache/` | cache for embeddings, extractions and data |

Embeddings and extractions are cached on disk by content hash, so re-running
anything on the same text is free.

## Repository layout

| path | what is there |
|---|---|
| `src/ledger.ts` | the claim ledger: append-only, with retract, `same_as` and `not_same_as` claims |
| `src/extract.ts` | claim extraction from a conversation session (one LLM call per session) |
| `src/graph.ts` | projection from claims to an entity graph, and Personalized PageRank |
| `src/facets.ts` | coarse facet tags on claims, and the one-call query planner |
| `src/retrieve.ts` | the read paths: dense, denseX, claim search, facet walk, RRF |
| `src/diff.ts` | graph diff between two extractions of the same sources |
| `src/llm.ts` | chat and embedding clients, with the on-disk cache |
| `bench/e*.ts` | the proof-of-concept experiments (E0–E6) |
| `bench/h2h/` | the head-to-head harness: adapters for every system, setup and run scripts |
| `results/` | raw result tables and the frozen configuration of each experiment |
| `docs/` | design, experiment log, results, literature notes |

## Reproducing the benchmarks

[BENCHMARKS.md](BENCHMARKS.md) walks through setting up every system and
running both head-to-heads. The datasets are not in this repository: LoCoMo
is CC BY-NC 4.0 and LongMemEval-S is on Hugging Face, and you download both
yourself.

## Status and limitations

This is a proof of concept, not a library you should depend on yet.

- **Storage.** The ledger is JSON lines and the graph lives in memory. The
  design targets Postgres (range types for the two time axes); that is not
  built.
- **API.** The only service interface is the Mem0-compatible REST server in
  `bench/h2h/`, which has no authentication or persistence. The library API
  sketched in [docs/CONCEPT.md](docs/CONCEPT.md) is not built.
- **Deletion.** Retraction is a new claim. A true purge (for the right to be
  forgotten) needs a separate, audited path, which does not exist yet.
- **Access control** is deliberately out of scope: a host system passes a
  filter (a "lens") into projection.
- **Graph traversal** in the HippoRAG style did not help in our tests (see
  RESULTS.md); what helps is using claims as retrieval keys.

## Related work

- [Mem0](https://github.com/mem0ai/mem0): extracts short facts from
  conversations and updates them in place.
- [Graphiti](https://github.com/getzep/graphiti): a temporal knowledge graph
  with validity windows on edges; the engine behind Zep.
- [Cognee](https://github.com/topoteretes/cognee): builds a knowledge graph
  and vector index from documents and conversations.
- [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG): Personalized
  PageRank over a passage and entity graph, strong on multi-hop QA.
- [GraphRAG](https://github.com/microsoft/graphrag): community summaries over
  an extracted entity graph.

How they differ from this design: [docs/research/landscape.md](docs/research/landscape.md).
Papers we read: [docs/research/papers.md](docs/research/papers.md).

## License

[Apache-2.0](LICENSE). The datasets used in the benchmarks have their own
licenses (LoCoMo is CC BY-NC 4.0) and are not included.
