// E2: does bitemporal slicing fix "stale version first"?
// Fixture: a private set of supersession timelines (38 timelines, 83
// revisions, 76 questions, 7 backfilled) from arlo-memory's benchmark suite.
// It is not part of this repo and is not public, so E2 cannot be reproduced
// from this repo alone; it is read from .cache/fixtures/supersession.json.
// Retrieval-level metric over the pooled 83 revisions:
//   top1      the expected revision is rank 1 of the whole pool
//   staleFirst another revision of the SAME timeline outranks the expected one
// Kill criterion: OGM no better than dense + as-of filter + recency blend.
// Held-out = timelines at index 2, 5, 8 of every ten (the fixture's split);
// tune only on the rest.
//
//   node bench/e2-currency.ts --tau 0.8

import { readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { extractSession } from "../src/extract.ts"
import { normEntity } from "../src/graph.ts"
import type { Claim } from "../src/ledger.ts"
import { CACHE, dot, embedMany, mapLimit } from "../src/llm.ts"
import type { Turn } from "../src/locomo.ts"

const { values: a } = parseArgs({ options: { docw: { type: "string", default: "0" }, recency: { type: "string", default: "0" }, tau: { type: "string", default: "0.8" }, slot: { type: "string", default: "similar" }, model: { type: "string", default: "claude-haiku-5-5" }, label: { type: "string", default: "v1" } } })
interface Rev { ref: string; title: string; content: string; occurredAt: string }
interface Q { id: string; kind: "current" | "as_of"; question: string; asOf?: string; expectRef: string }
interface Tl { id: string; revisions: Rev[]; questions: Q[]; backfilled?: boolean }
const TL = JSON.parse(readFileSync(join(CACHE, "fixtures", "supersession.json"), "utf8")) as Tl[]
const heldOut = (i: number) => [2, 5, 8].includes(i % 10)
const docs = TL.flatMap((t, ti) => t.revisions.map((r) => ({ id: `${t.id}/${r.ref}`, tl: t.id, ti, text: `${r.title}\n${r.content}`, date: r.occurredAt })))
const log = (m: string) => console.log(`[e2 ${new Date().toISOString().slice(11, 19)}] ${m}`)

// Ledger: one extraction per revision, the revision as a one-turn "session"
// dated by its occurredAt. A claim's valid time is what the TEXT says
// ("effective 1 October") when it says it, else the revision date.
const claimsPer = await mapLimit(docs, 3, (d) => {
  const turn: Turn = { id: "D1:1", session: d.id, sessionIndex: 0, date: new Date(`${d.date}T12:00:00Z`), speaker: "doc", text: d.text }
  return extractSession(a.model!, 0, d.id, turn.date, d.date, [turn])
})
const claims: Array<Claim & { doc: string; valid: string }> = claimsPer.flatMap((cs, i) => cs.map((c) => ({ ...c, doc: docs[i]!.id, valid: (c.validFrom && /^\d{4}-\d{2}-\d{2}/.test(c.validFrom) ? c.validFrom.slice(0, 10) : docs[i]!.date) })))
log(`${docs.length} revisions → ${claims.length} claims`)

const docVec = await embedMany(docs.map((d) => d.text))
const claimVec = await embedMany(claims.map((c) => c.text))
const qs = TL.flatMap((t, ti) => t.questions.map((q) => ({ ...q, tl: t.id, ti, backfilled: !!t.backfilled })))
const qVec = await embedMany(qs.map((q) => q.question))
const tau = Number(a.tau)

type Ranker = (qi: number) => string[]
const asOfOf = (qi: number) => qs[qi]!.asOf ?? "9999-12-31"
const byScore = (s: Map<string, number>) => [...s.entries()].sort((x, y) => y[1] - x[1]).map(([id]) => id)

const methods: Record<string, Ranker> = {
  "dense": (qi) => byScore(new Map(docs.map((d, i) => [d.id, dot(qVec[qi]!, docVec[i]!)]))),
  "dense+asOf": (qi) => byScore(new Map(docs.filter((d) => d.date <= asOfOf(qi)).map((d) => [d.id, dot(qVec[qi]!, docVec[docs.indexOf(d)]!)]))),
  // Literature baseline (Cognis-style 0.6 relevance / 0.4 recency), after the as-of filter.
  "dense+asOf+recency": (qi) => {
    const pool = docs.map((d, i) => ({ d, s: dot(qVec[qi]!, docVec[i]!) })).filter((x) => x.d.date <= asOfOf(qi))
    const dates = [...new Set(pool.map((x) => x.d.date))].sort()
    const lo = Math.min(...pool.map((x) => x.s)), hi = Math.max(...pool.map((x) => x.s))
    return byScore(new Map(pool.map((x) => [x.d.id, 0.6 * ((x.s - lo) / (hi - lo || 1)) + 0.4 * (dates.indexOf(x.d.date) / Math.max(1, dates.length - 1))])))
  },
  // OGM: claims ranked by similarity; the top claim defines a SLOT (claims
  // about the same subject saying the same kind of thing, cosine >= tau);
  // the slot is resolved on VALID time (latest valid <= as-of), so a
  // backfilled old memo or a more-similar old phrasing cannot win.
  "ogm(bitemporal)": (qi) => {
    const asOf = asOfOf(qi)
    // Optional recency blend on slot HEADS (0 = similarity only): heads compete
    // across topics, so a little recency pushes the live topic's slot up.
    const rw = Number(a.recency)
    const days = claims.map((c) => Date.parse(c.valid))
    const lo = Math.min(...days), hi = Math.max(...days)
    // docw: blend in the source document's similarity, so an atomic claim
    // keeps its context (a "per seat" claim from the budget memo vs the price memo).
    const dw = Number(a.docw)
    const docSim = new Map(docs.map((d, k) => [d.id, dot(qVec[qi]!, docVec[k]!)]))
    const ranked = claims.map((c, i) => ({ c, i, s: (1 - dw) * dot(qVec[qi]!, claimVec[i]!) + dw * docSim.get(c.doc)! + rw * ((days[i]! - lo) / (hi - lo || 1)) * (c.valid <= asOf ? 1 : 0) })).sort((x, y) => y.s - x.s)
    const out: string[] = []
    const used = new Set<number>()
    for (const head of ranked) {
      if (used.has(head.i)) continue
      const subj = normEntity(head.c.subject)
      const slot = ranked.filter((x) => !used.has(x.i) && (x.i === head.i || ((a.slot === "similar" || normEntity(x.c.subject) === subj) && dot(claimVec[x.i]!, claimVec[head.i]!) >= tau)))
      slot.forEach((x) => used.add(x.i))
      const valid = slot.filter((x) => x.c.valid <= asOf).sort((x, y) => y.c.valid.localeCompare(x.c.valid) || y.s - x.s)
      for (const x of [...valid, ...slot.filter((x) => x.c.valid > asOf)]) if (!out.includes(x.c.doc)) out.push(x.c.doc)
      if (out.length >= docs.length) break
    }
    for (const d of docs) if (!out.includes(d.id)) out.push(d.id)
    return out
  },
}

if (process.env.E2_DEBUG) {
  const qi = qs.findIndex((q) => q.id === process.env.E2_DEBUG)
  const ranked = claims.map((c, i) => ({ c, i, s: dot(qVec[qi]!, claimVec[i]!) })).sort((x, y) => y.s - x.s).slice(0, 8)
  for (const x of ranked) console.log(x.s.toFixed(3), x.c.doc, x.c.valid, normEntity(x.c.subject), "|", x.c.text.slice(0, 80), "| cos-to-top", dot(claimVec[x.i]!, claimVec[ranked[0]!.i]!).toFixed(2))
  console.log(methods["ogm(bitemporal)"]!(qi).slice(0, 5))
}
const table: Record<string, Record<string, unknown>> = {}
for (const [name, rank] of Object.entries(methods)) {
  const rows = qs.map((q, qi) => {
    const r = rank(qi)
    const exp = `${q.tl}/${q.expectRef}`
    const er = r.indexOf(exp)
    const stale = r.findIndex((id) => id.startsWith(`${q.tl}/`) && id !== exp)
    if (process.env.E2_FAILS === name && er !== 0) console.log(`FAIL ${q.id} exp=${exp} got=${r.slice(0, 3).join(",")} | ${q.question}`)
    return { q, top1: er === 0, staleFirst: er < 0 || (stale >= 0 && stale < er) }
  })
  const slice = (f: (x: (typeof rows)[number]) => boolean) => {
    const l = rows.filter(f)
    return { n: l.length, top1: +((100 * l.filter((x) => x.top1).length) / l.length).toFixed(1), staleFirst: +((100 * l.filter((x) => x.staleFirst).length) / l.length).toFixed(1) }
  }
  table[name] = { all: slice(() => true), tune: slice((x) => !heldOut(x.q.ti)), heldOut: slice((x) => heldOut(x.q.ti)), current: slice((x) => x.q.kind === "current"), asOf: slice((x) => x.q.kind === "as_of"), backfilled: slice((x) => x.q.backfilled) }
}
writeFileSync(join(import.meta.dirname, "..", "results", `e2-${a.label}.json`), JSON.stringify({ experiment: "E2", ts: new Date().toISOString(), tau, claims: claims.length, table }, null, 2))
console.log(`method                | slice: top1 / staleFirst (lower better)`)
for (const [m, t] of Object.entries(table)) console.log(`${m.padEnd(21)} | ${Object.entries(t).map(([s, v]) => `${s} ${(v as { top1: number }).top1}/${(v as { staleFirst: number }).staleFirst}`).join(" | ")}`)
