// LongMemEval-S (xiaowu0162/longmemeval-cleaned), seeded subset: 15 questions
// each of multi-session, temporal-reasoning, knowledge-update (subset ids in
// results/lme-seen-ids.json, picked with random.Random(7) before any run).
// Data stays in .cache. Standard session-level metric: recall_any@k / recall_all@k
// over the question's haystack (~50 sessions), k = 1, 3, 5, 10.
//
// Units: one per USER turn, rendered with the assistant reply that follows,
// truncated to 300 chars (LME evidence is user-stated; assistant replies are
// long and mostly filler). The extractor sees the same rendering.
//
//   node bench/lme-recall.ts --phase extract   (ledger for all subset sessions)
//   node bench/lme-recall.ts --phase eval

import { readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { extractSession, EXTRACTOR_VERSION } from "../src/extract.ts"
import { planQuery, tagClaims, type QueryPlan } from "../src/facets.ts"
import { project } from "../src/graph.ts"
import type { Claim } from "../src/ledger.ts"
import { CACHE, embedMany, mapLimit } from "../src/llm.ts"
import type { Turn } from "../src/locomo.ts"
import { claimsRank, dense, denseX, facetRank, rrf, type Index, type Ranking } from "../src/retrieve.ts"

const { values: a } = parseArgs({ options: { phase: { type: "string", default: "eval" }, model: { type: "string", default: "claude-haiku-5-5" }, concurrency: { type: "string", default: "6" }, label: { type: "string", default: "subset45" } } })
const log = (m: string) => console.log(`[lme ${new Date().toISOString().slice(11, 19)}] ${m}`)
interface Msg { role: "user" | "assistant"; content: string; has_answer?: boolean }
const evidenceUnits = new Map<string, Set<string>>() // question id → unit ids of user turns flagged has_answer
interface Item { question_id: string; question_type: string; question: string; question_date: string; answer: string; answer_session_ids: string[]; haystack_dates: string[]; haystack_session_ids: string[]; haystack_sessions: Msg[][] }
const ids = new Set(JSON.parse(readFileSync(join(import.meta.dirname, "..", "results", "lme-seen-ids.json"), "utf8")) as string[])
const items = (JSON.parse(readFileSync(join(CACHE, "lme", "longmemeval_s_cleaned.json"), "utf8")) as Item[]).filter((x) => ids.has(x.question_id))

function lmeDate(s: string): Date {
  const m = /^(\d{4})\/(\d{2})\/(\d{2}) \(\w+\) (\d{2}):(\d{2})/.exec(s)
  if (!m) throw new Error(`bad LME date ${s}`)
  return new Date(Date.UTC(+m[1]!, +m[2]! - 1, +m[3]!, +m[4]!, +m[5]!))
}

/** User-turn units of one session; ids "<session>#<n>". */
function units(sid: string, date: Date, msgs: Msg[]): Turn[] {
  const out: Turn[] = []
  let prev: string | undefined
  msgs.forEach((m, i) => {
    if (m.role !== "user") return
    const reply = msgs[i + 1]?.role === "assistant" ? msgs[i + 1]!.content.slice(0, 300) : ""
    const text = `User: ${m.content}${reply ? `\nAssistant: ${reply}` : ""}`
    out.push({ id: `${sid}#${out.length}`, session: sid, sessionIndex: 0, date, speaker: "User", text, ...(prev ? { prev } : {}) })
    prev = text
  })
  return out
}

for (const x of items) {
  const ev = new Set<string>()
  x.haystack_session_ids.forEach((sid, i) => {
    let n = 0
    for (const m of x.haystack_sessions[i]!) { if (m.role !== "user") continue; if (m.has_answer) ev.add(`${sid}#${n}`); n++ }
  })
  evidenceUnits.set(x.question_id, ev)
}
const sessions = new Map<string, { date: Date; turns: Turn[] }>()
for (const x of items) x.haystack_session_ids.forEach((sid, i) => {
  if (!sessions.has(sid)) { const date = lmeDate(x.haystack_dates[i]!); sessions.set(sid, { date, turns: units(sid, date, x.haystack_sessions[i]!) }) }
})
log(`${items.length} questions, ${sessions.size} sessions, extractor ${EXTRACTOR_VERSION}`)

let done = 0
const ledger = new Map<string, Claim[]>()
await mapLimit([...sessions.entries()], Number(a.concurrency), async ([sid, s]) => {
  if (!s.turns.length) { ledger.set(sid, []); return }
  // A few haystack sessions are enormous; cap the transcript so one call stays bounded.
  const turns = s.turns.map((t) => ({ ...t, text: t.text.slice(0, 2500) }))
  try { ledger.set(sid, await extractSession(a.model!, 0, sid, s.date, s.date.toISOString(), turns)) } catch (e) { log(`extract failed ${sid}: ${String(e).slice(0, 120)}`); ledger.set(sid, []) }
  if (++done % 100 === 0) log(`extracted ${done}/${sessions.size}`)
})
if (a.phase === "extract") process.exit(0)

const KS = [1, 3, 5, 10]
type M = (ix: Index, q: Float32Array, b: { dense: Ranking; plan: QueryPlan; tags: Map<string, string[]> }) => Ranking
const METHODS: Record<string, M> = {
  "dense": (_ix, _q, b) => b.dense,
  "denseX": (ix, q) => denseX(ix, q),
  "rrf(dense,claims)": (ix, q, b) => rrf([b.dense, claimsRank(ix, q, b.dense)]),
  "rrf(dense,claims,facet)": (ix, q, b) => rrf([b.dense, claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "rrf(dense,denseX,claims,facet)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
}
const TK = [5, 10, 20]
const rows: Array<{ type: string; m: string; any: number[]; all: number[]; tany: number[]; tall: number[]; tn: number }> = []
for (const [qi, x] of items.entries()) {
  const turns = x.haystack_session_ids.flatMap((sid) => sessions.get(sid)!.turns)
  const claims = x.haystack_session_ids.flatMap((sid) => ledger.get(sid) ?? [])
  const byTurn = new Map<string, string[]>()
  for (const c of claims) for (const s of c.sources) byTurn.set(s, [...(byTurn.get(s) ?? []), c.text])
  const turnVec = await embedMany(turns.map((t) => (t.prev ? `${t.prev}\n${t.text}` : t.text)))
  const turnVecX = await embedMany(turns.map((t) => `${t.prev ? `${t.prev}\n` : ""}${t.text}${byTurn.has(t.id) ? `\nFacts: ${byTurn.get(t.id)!.join(" ")}` : ""}`))
  const claimVec = await embedMany(claims.map((c) => `${c.text}${c.validFrom ? ` (${c.validFrom})` : ""}`))
  const ix: Index = { turns, turnVec, turnVecX, claims, claimVec, graph: await project(claims, turns, { synonymThreshold: 1, turnAdjacency: 0 }), pos: new Map(turns.map((t, i) => [t.id, i])) }
  const [qv] = await embedMany([x.question])
  const tags = await tagClaims(a.model!, claims)
  const plan = await planQuery(a.model!, x.question, ["the user", "the assistant"])
  const base = { dense: dense(ix, qv!), plan, tags }
  const gold = new Set(x.answer_session_ids)
  for (const [m, f] of Object.entries(METHODS)) {
    const ranking = f(ix, qv!, base)
    const ev = evidenceUnits.get(x.question_id)!
    const tany = TK.map((k) => (ev.size && ranking.slice(0, k).some((u) => ev.has(u)) ? 1 : 0))
    const tall = TK.map((k) => (ev.size && [...ev].every((u) => ranking.slice(0, k).includes(u)) ? 1 : 0))
    const sess: string[] = []
    for (const id of ranking) { const s = id.slice(0, id.lastIndexOf("#")); if (!sess.includes(s)) sess.push(s) }
    rows.push({ type: x.question_type, m, any: KS.map((k) => (sess.slice(0, k).some((s) => gold.has(s)) ? 1 : 0)), all: KS.map((k) => ([...gold].every((g) => sess.slice(0, k).includes(g)) ? 1 : 0)), tany, tall, tn: ev.size })
  }
  if ((qi + 1) % 5 === 0) log(`evaluated ${qi + 1}/${items.length}`)
}
const table: Record<string, Record<string, Record<string, number>>> = {}
for (const m of Object.keys(METHODS)) {
  table[m] = {}
  for (const t of ["all", "multi-session", "temporal-reasoning", "knowledge-update"]) {
    const l = rows.filter((r) => r.m === m && (t === "all" || r.type === t))
    const lt = l.filter((r) => r.tn > 0)
    table[m]![t] = Object.fromEntries([...TK.map((k, i) => [`turn any@${k}`, +((100 * lt.reduce((s, r) => s + r.tany[i]!, 0)) / lt.length).toFixed(1)]), ...TK.map((k, i) => [`turn all@${k}`, +((100 * lt.reduce((s, r) => s + r.tall[i]!, 0)) / lt.length).toFixed(1)]), ["turn n", lt.length],...KS.map((k, i) => [`any@${k}`, +((100 * l.reduce((s, r) => s + r.any[i]!, 0)) / l.length).toFixed(1)]), ...KS.map((k, i) => [`all@${k}`, +((100 * l.reduce((s, r) => s + r.all[i]!, 0)) / l.length).toFixed(1)]), ["n", l.length]])
  }
}
writeFileSync(join(import.meta.dirname, "..", "results", `lme-${a.label}.json`), JSON.stringify({ experiment: "LongMemEval-S subset", ts: new Date().toISOString(), extractor: EXTRACTOR_VERSION, questions: items.length, sessions: sessions.size, claims: [...ledger.values()].reduce((s, l) => s + l.length, 0), table }, null, 2))
for (const [m, t] of Object.entries(table)) console.log(`${m.padEnd(32)} ${Object.entries(t).map(([s, v]) => `${s.slice(0, 9)} turn all@10 ${v["turn all@10"]} all@20 ${v["turn all@20"]}`).join(" | ")}`)
