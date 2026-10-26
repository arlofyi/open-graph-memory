// E6: do recall gains become ANSWER gains at a small context?
// LoCoMo TEST, top-k turns (default 10) from each method → one answerer call
// → one judge call (CORRECT / WRONG, lenient on wording, strict on facts).
// Prompts are this repo's own (NOT Mem0's), so absolute numbers are not
// comparable with numbers from other harnesses; the comparison is
// between methods under identical prompts, models and k.
//
//   node bench/e6-qa.ts --k 10 --tag haiku2

import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { planQuery, tagClaims } from "../src/facets.ts"
import { project } from "../src/graph.ts"
import { Ledger } from "../src/ledger.ts"
import { CACHE, chat, embedMany, mapLimit, parseJson } from "../src/llm.ts"
import { CATEGORY, TEST, loadLocomo } from "../src/locomo.ts"
import { claimsRank, dense, denseX, facetRank, rrf, type Index, type Ranking } from "../src/retrieve.ts"

const { values: a } = parseArgs({ options: { k: { type: "string", default: "10" }, tag: { type: "string", default: "haiku2" }, model: { type: "string", default: "claude-haiku-5-5" }, convs: { type: "string", default: TEST.join(",") }, concurrency: { type: "string", default: "2" } } })
const K = Number(a.k)
const data = loadLocomo()
const log = (m: string) => console.log(`[e6 ${new Date().toISOString().slice(11, 19)}] ${m}`)

const ANSWER = `You answer a question about two people's long chat history using ONLY the memories given (each is one chat message with its date). Resolve relative dates ("last week") against the message date. Answer in a short phrase (under 15 words). If the memories do not contain the answer, say "unknown".`
const JUDGE = `Grade an answer against the gold answer. CORRECT if it states the same fact(s) (wording, extra detail or format may differ; for dates, the same day/month/year as asked; for lists, the main items). WRONG if it misses, contradicts, or says unknown. JSON {"label":"CORRECT"|"WRONG"}`

type M = (ix: Index, q: Float32Array, b: { dense: Ranking; plan: Awaited<ReturnType<typeof planQuery>>; tags: Map<string, string[]> }) => Ranking
const METHODS: Record<string, M> = {
  "dense": (_ix, _q, b) => b.dense,
  "R2-A rrf(dense,denseX,claims,facet)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "R2-B denseX": (ix, q) => denseX(ix, q),
}
const rows: Array<{ m: string; category: number; ok: boolean }> = []
for (const ci of a.convs!.split(",").map(Number)) {
  const conv = data[ci]!
  const claims = new Ledger(join(CACHE, "ledger", a.tag!, `conv${ci}.jsonl`)).live()
  const byTurn = new Map<string, string[]>()
  for (const c of claims) for (const s of c.sources) byTurn.set(s, [...(byTurn.get(s) ?? []), c.text])
  const ix: Index = {
    turns: conv.turns,
    turnVec: await embedMany(conv.turns.map((t) => (t.prev ? `${t.prev}\n${t.text}` : t.text))),
    turnVecX: await embedMany(conv.turns.map((t) => `${t.prev ? `${t.prev}\n` : ""}${t.text}${byTurn.has(t.id) ? `\nFacts: ${byTurn.get(t.id)!.join(" ")}` : ""}`)),
    claims,
    claimVec: await embedMany(claims.map((c) => `${c.text}${c.validFrom ? ` (${c.validFrom}${c.validTo ? ` to ${c.validTo}` : ""})` : ""}`)),
    graph: await project(claims, conv.turns, { synonymThreshold: 1, turnAdjacency: 0 }),
    pos: new Map(conv.turns.map((t, i) => [t.id, i])),
  }
  const tags = await tagClaims(a.model!, claims)
  const qs = conv.questions
  const qv = await embedMany(qs.map((q) => q.question))
  const plans = await mapLimit(qs, 2, (q) => planQuery(a.model!, q.question, conv.speakers))
  const jobs = qs.flatMap((q, qi) => Object.keys(METHODS).map((m) => ({ q, qi, m })))
  const res = await mapLimit(jobs, Number(a.concurrency), async ({ q, qi, m }) => {
    const top = METHODS[m]!(ix, qv[qi]!, { dense: dense(ix, qv[qi]!), plan: plans[qi]!, tags }).slice(0, K)
    const mem = top.map((id) => conv.turns[ix.pos.get(id)!]!).sort((x, y) => x.date.getTime() - y.date.getTime()).map((t) => `[${t.date.toISOString().slice(0, 10)}] ${t.text}`).join("\n")
    const ans = (await chat(a.model!, [{ role: "system", content: ANSWER }, { role: "user", content: `Memories:\n${mem}\n\nQuestion: ${q.question}` }], { maxTokens: 120 })).trim()
    const v = parseJson<{ label: string }>(await chat(a.model!, [{ role: "system", content: JUDGE }, { role: "user", content: `Question: ${q.question}\nGold: ${q.answer}\nAnswer: ${ans}` }], { json: true, maxTokens: 60 }))
    return { m, category: q.category, ok: v.label === "CORRECT" }
  })
  rows.push(...res)
  log(`conv ${ci}: ${qs.length} questions × ${Object.keys(METHODS).length} methods judged`)
}
const pct = (l: typeof rows) => +((100 * l.filter((r) => r.ok).length) / l.length).toFixed(1)
const table = Object.fromEntries(Object.keys(METHODS).map((m) => {
  const l = rows.filter((r) => r.m === m)
  return [m, { all: pct(l), n: l.length, ...Object.fromEntries(Object.entries(CATEGORY).map(([c, name]) => [name, pct(l.filter((r) => r.category === Number(c)))])) }]
}))
writeFileSync(join(import.meta.dirname, "..", "results", `e6-qa-k${K}.json`), JSON.stringify({ experiment: "E6 judged QA", ts: new Date().toISOString(), k: K, answerer: a.model, judge: a.model, ledger: a.tag, table }, null, 2))
console.log(JSON.stringify(table, null, 1))
