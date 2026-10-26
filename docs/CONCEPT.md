# Concept: open-graph-memory

*This is the design the proof of concept tests. Parts of it are not built yet:
the PoC keeps the ledger as JSON lines and the graph in memory, and the API in
§6 is a sketch. Where the experiments changed the design, it says so.*

## 1. Thesis

The knowledge graph should not be the database. **The claim ledger is the
database; the graph is a view.**

Four commitments follow, and each is a design constraint rather than a feature:

1. **Append-only.** Nothing is updated or deleted in the ledger. Corrections are
   new claims that *retract* or *supersede* older ones. (Hard delete for
   right-to-be-forgotten is a separate, audited purge path.)
2. **Every claim is sourced.** No source, no claim. Sources are spans in an
   ingested document, conversation turn, API record, or another claim.
3. **Every claim is bitemporal.** *Valid time* (when it was true in the world)
   and *recorded time* (when we learned it) are separate axes.
4. **Every derived thing is reproducible.** An entity profile, an edge, or a
   merge decision is the output of `project(ledger, lens, projector@version)`.
   Same inputs give the same graph.

## 2. Data model

Four tables. Postgres first (pgvector plus range types), SQLite later. No graph
database: adjacency in Postgres is enough at the scales this targets.

### 2.1 `source`
Immutable pointer to evidence: `{id, kind: document|turn|record|claim, uri,
span?, content_hash, loaded_at, authored_at?, actor}`. `uri` points back into the
host system (a document chunk, a message id), so citations round-trip.

### 2.2 `claim` (the ledger)

```
claim
  id              uuid
  subject         mention_id         -- what the claim is about
  predicate       text               -- open vocabulary; normalised later (§4)
  object          mention_id | literal jsonb
  qualifiers      jsonb              -- "in 2019", "for $10M", role, unit
  text            text               -- canonical standalone sentence
  valid_time      tstzrange          -- when true in the world (may be open/unknown)
  recorded_time   tstzrange          -- [learned_at, retracted_at)
  source_ids      uuid[]             -- ≥1, enforced
  confidence      real [0,1]
  actor_type      user|machine|ai
  extractor       text               -- e.g. "claims@3 / gpt-oss-120b"
  kind            assert|retract|same_as|not_same_as
  retracts        uuid?              -- for kind=retract
```

Ideas taken from HASH: the claim is the atomic, standalone, single-predicate
sentence. **Qualifiers are kept separate from the triple**, because they become
edge properties. Separate valid and recorded axes with a GiST `EXCLUDE` against
overlapping rectangles. `actor_type = ai` as a first-class actor.

Departure from HASH: in HASH, claims feed an *entity store* that is then
mutated. Here the claims are the only store.

### 2.3 `mention`
A surface reference: `{id, text, type_hint, source_id, embedding}`. "Nintendo",
"NoE" and "the Kyoto company" are three mentions. **Mentions never merge.**
Only projections cluster them into entities.

### 2.4 `projection` (materialised, disposable)
`{projection_id, lens_hash, projector_version, built_at}` plus
`entity(id, mention_ids[], label, profile jsonb)` and
`edge(src, dst, predicate, qualifiers, valid_time, claim_ids[])`.

Every entity and edge carries the `claim_ids` that justify it. HASH did the same
per property value with `claimIdsUsedToDetermineValue`, and it is the single
best idea in that codebase. Drop the projection tables and rebuild them at any time.

## 3. Projection: from claims to graph

`project(ledger, lens, projector)` runs in four deterministic passes:

1. **Filter by lens.** The lens is an injected predicate over `source` and `claim`
   (`(claim) => boolean` plus an equivalent SQL fragment). The host system
   compiles its own access policy (clearance, domain, owner) into this.
   open-graph-memory has **zero** built-in access control. Compiling a policy
   into a filter is an idea from HASH's use of Cedar; the policy engine itself
   belongs to the host.
2. **Resolve mentions to entities.** First apply explicit `same_as` and
   `not_same_as` claims; they always win. Then vector-recall candidates on label
   embeddings (same type hint), then a conservative adjudicator ("if in doubt,
   don't merge"; regional subsidiaries and product versions are *not* duplicates).
   **The adjudicator's decisions are written back as `same_as` claims with
   `actor_type = ai`.** They are now reviewable, retractable and replayable.
3. **Normalise predicates** onto a small, growable vocabulary (§4).
4. **Build the temporal state.** For each `(entity, predicate)`, compute the
   timeline of valid intervals. "Current" means the open valid interval as of now
   in recorded time. Contradictions with overlapping valid time are kept as
   **conflict** edges and exposed to the caller, never silently resolved.

### Graph diff
Projecting the same ledger with `projector@v` and `projector@v+1` gives a
structural diff: merges added or removed, edges gained or lost, timelines changed.
This is both a user-facing feature ("what changed in my graph when we upgraded
extraction") and the regression harness for every extractor change.

## 4. Ontology: open-world on purpose

HASH requires a globally URL-addressed type for every property. That is
powerful, but it is a lot to ask before the first fact can be stored. Instead:

- Predicates start **free-text** and are clustered into a vocabulary by
  embedding plus usage counts. A predicate graduates to a "typed" predicate (with
  domain/range hints and an inverse name) once it is used often enough or a human
  pins it.
- Types are **optional schema packs**, versioned `…/v/N` (SemType-style URLs,
  MIT/Apache-licensed, portable). A pack *constrains projection*; it never gates
  ingest. Unknown predicates still land in the ledger.
- Prompts see short type ids (`t3`, `p17`) that are mapped back afterwards (a
  HASH prompt trick, re-derived rather than copied).

## 5. Retrieval: evidence sets, not top-k

The goal is **completeness**: return *all* the claims needed to answer. arlo-memory's
retriever on LoCoMo finds *one* piece of evidence about 94% of the time in
its top 50, but *all* of it only about 83% of the time, and in its top 10 that
falls to 66%. That gap is the reason a graph layer exists.

*What the experiments changed:* step 2 below (PageRank spreading) added nothing
on LoCoMo. What worked was using claims as retrieval keys, fused with dense
retrieval, plus a typed facet traversal. See RESULTS.md.

1. **Anchor.** Link the question's mentions to entities in the projection
   (vector plus lexical). Fall back to dense retrieval if nothing links.
2. **Spread.** Run Personalized PageRank from the anchors over the bipartite
   *claim ↔ entity* graph (HippoRAG 2's best result, applied to claims rather than
   passages). Use a bounded traversal spec with a depth per predicate class (a
   HASH idea) so expansion stays predictable.
3. **Time-slice.** Apply `as_of_valid` / `as_of_recorded` before ranking. This
   fixes the "older version ranked above the current one" failure at the data
   layer, not with a reranker.
4. **Return an evidence set.** The result is
   `{claims[], entities[], conflicts[], timelines[], coverage}`. `coverage` is
   a self-estimate ("2 of 3 anchors have supporting claims"), so callers can ask
   again or say "I don't know".

The output plugs into any answerer. Because every claim carries `source.uri`,
an answerer can cite the original messages or document chunks.

## 6. API surface (sketch)

```ts
const ogm = createGraphMemory({ db, embedder, extractor, lens });

await ogm.ingest(source)                    // extract → claims (append-only)
await ogm.assert(claim) / ogm.retract(id)   // manual / human writes
await ogm.sameAs(a, b, why) / ogm.notSameAs(a, b, why)

const g = await ogm.project({ lens, asOf })        // cached, disposable
const ev = await ogm.evidence(question, { asOf })  // §5
const tl = await ogm.timeline(entity, predicate?)  // valid-time history
const d  = await ogm.diff(projectorA, projectorB)  // graph diff
```

MCP tools: `graph_evidence`, `graph_timeline`, `graph_assert`, `graph_retract`,
`graph_same_as`. Five tools, read-heavy.

## 7. Explicit non-goals

- No query language. A JSON filter AST plus `evidence()` is enough.
- No governance, encryption or audit. Those belong to the host system.
- No UI blocks or Block Protocol.
- No graph database, no Temporal.io, no Vault. One Postgres instance.
- No "global community summaries" (GraphRAG-global). The evidence says they score
  badly on temporal questions.

## 8. Open questions

- Can claim extraction work on **chat**? Fact extractors built for documents
  pull almost nothing out of conversation. If claim extraction on conversation
  is also weak, the ledger is empty exactly where agents need it. This was
  experiment E0 and the first kill rule. (Answer: yes, at 84% precision with
  one provenance rule.)
- Cost: extraction plus adjudication per source. Is gpt-oss-20b good enough for
  claims, so that only adjudication needs a bigger model?
- How expensive is projection at about 1M claims? Is incremental projection
  (only the dirty entities) good enough, or is materialisation the bottleneck?
- Is "undo a merge" something users actually ask for, or only something
  engineers want?
