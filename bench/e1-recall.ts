// E1: does PPR over the claim graph close the "all evidence" gap?
// LoCoMo evidence recall: for each scored question, where do ALL its evidence
// turns land in the ranking? any@k / all@k, per split and category.
//
//   node bench/e1-recall.ts --convs 0,1 --label dev
//
// Iterate on DEV (convs 0-1) only. TEST (2-9) is run once per frozen config.

import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { project, type GraphOptions } from "../src/graph.ts"
import { Ledger } from "../src/ledger.ts"
import { CACHE, embedMany, mapLimit } from "../src/llm.ts"
import { CATEGORY, DEV, loadLocomo, type Question } from "../src/locomo.ts"
import { claimsRank, dense, denseX, facetRank, pprRank, rrf, withNext, type Index, type PprOptions, type Ranking } from "../src/retrieve.ts"
import { planQuery, tagClaims, type QueryPlan } from "../src/facets.ts"

const { values: a } = parseArgs({ options: {
  convs: { type: "string", default: "0,1" },
  label: { type: "string", default: "dev" },
  methods: { type: "string", default: "" },
  synonym: { type: "string", default: "0.8" },
  adjacency: { type: "string", default: "0" },
  model: { type: "string", default: "claude-haiku-5-5" },
  tag: { type: "string", default: "" },
} })
const convs = a.convs!.split(",").map(Number)
const KS = [10, 20, 50]
const log = (m: string) => console.log(`[e1 ${new Date().toISOString().slice(11, 19)}] ${m}`)
const data = loadLocomo()

const HIPPO: PprOptions = { topClaims: 5, passageWeight: 0.05, specificity: false, damping: 0.5 }
interface Base { dense: Ranking; plan: QueryPlan; tags: Map<string, string[]> }
type Method = (ix: Index, q: Float32Array, base: Base) => Ranking
const METHODS: Record<string, Method> = {
  "dense": (_ix, _q, b) => b.dense,
  "dense+next": (ix, _q, b) => withNext(ix, b.dense),
  "claims": (ix, q, b) => claimsRank(ix, q, b.dense),
  "ppr(hippo2)": (ix, q) => pprRank(ix, q, HIPPO),
  "ppr(top20)": (ix, q) => pprRank(ix, q, { ...HIPPO, topClaims: 20 }),
  "ppr(top20,pw.3)": (ix, q) => pprRank(ix, q, { ...HIPPO, topClaims: 20, passageWeight: 0.3 }),
  "ppr(top20,spec)": (ix, q) => pprRank(ix, q, { ...HIPPO, topClaims: 20, specificity: true }),
  "rrf(dense,claims)": (ix, q, b) => rrf([b.dense, claimsRank(ix, q, b.dense)]),
  "rrf(dense,ppr)": (ix, q, b) => rrf([b.dense, pprRank(ix, q, { ...HIPPO, topClaims: 20 })]),
  "rrf(dense,claims,ppr)": (ix, q, b) => rrf([b.dense, claimsRank(ix, q, b.dense), pprRank(ix, q, { ...HIPPO, topClaims: 20 })]),
  "denseX": (ix, q) => denseX(ix, q),
  "rrf(denseX,claims,facet)": (ix, q, b) => rrf([denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "rrf(dense,denseX,claims,facet)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "route(agg→mmr.7)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense, b.plan.aggregate ? 0.7 : 0)]),
  "route(agg→mmr.5)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense, b.plan.aggregate ? 0.5 : 0)]),
  "rrf4(mmr.7 all)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense, 0.7)]),
  "route(agg→facet×2)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)], 60, [1, 1, 1, b.plan.aggregate ? 2 : 1]),
  "route(agg→facet×3)": (ix, q, b) => rrf([b.dense, denseX(ix, q), claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)], 60, [1, 1, 1, b.plan.aggregate ? 3 : 1]),
  "facet": (ix, q, b) => facetRank(ix, q, b.plan, b.tags, b.dense),
  "rrf(dense,facet)": (ix, q, b) => rrf([b.dense, facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "rrf(dense,claims,facet)": (ix, q, b) => rrf([b.dense, claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]),
  "route(agg→facet)": (ix, q, b) => (b.plan.aggregate ? rrf([b.dense, claimsRank(ix, q, b.dense), facetRank(ix, q, b.plan, b.tags, b.dense)]) : rrf([b.dense, claimsRank(ix, q, b.dense)])),
}
const chosen = a.methods ? a.methods.split(";") : Object.keys(METHODS)

interface Row { qid: string; conv: number; category: number; n: number; any: Record<string, number | null>; all: Record<string, number | null> }
const rows: Record<string, Row[]> = Object.fromEntries(chosen.map((m) => [m, []]))
const gopts: GraphOptions = { synonymThreshold: Number(a.synonym), turnAdjacency: Number(a.adjacency) }
const graphStats: Record<number, Record<string, number>> = {}

function ranks(evidence: string[], r: Ranking): { first: number | null; all: number | null } {
  const need = new Set(evidence)
  let first: number | null = null
  let all: number | null = null
  for (let i = 0; i < r.length && need.size; i++) if (need.delete(r[i]!)) { first ??= i + 1; if (!need.size) all = i + 1 }
  return { first, all }
}

for (const ci of convs) {
  const conv = data[ci]!
  const ledger = new Ledger(join(CACHE, "ledger", a.tag!, `conv${ci}.jsonl`))
  const claims = ledger.live()
  if (!claims.length) throw new Error(`conv ${ci}: empty ledger. Run E0 first.`)
  log(`conv ${ci}: ${conv.turns.length} turns, ${claims.length} claims; embedding`)
  const turnVec = await embedMany(conv.turns.map((t) => (t.prev ? `${t.prev}\n${t.text}` : t.text)), log)
  const claimVec = await embedMany(claims.map((c) => `${c.text}${c.validFrom ? ` (${c.validFrom}${c.validTo ? ` to ${c.validTo}` : ""})` : ""}`), log)
  const byTurn = new Map<string, string[]>()
  for (const c of claims) for (const s of c.sources) byTurn.set(s, [...(byTurn.get(s) ?? []), c.text])
  const turnVecX = chosen.some((m) => m.includes("denseX")) ? await embedMany(conv.turns.map((t) => `${t.prev ? `${t.prev}\n` : ""}${t.text}${byTurn.has(t.id) ? `\nFacts: ${byTurn.get(t.id)!.join(" ")}` : ""}`), log) : undefined
  const graph = await project(claims, conv.turns, gopts)
  graphStats[ci] = graph.stats
  log(`conv ${ci}: graph ${JSON.stringify(graph.stats)}`)
  const ix: Index = { turns: conv.turns, turnVec, turnVecX, claims, claimVec, graph, pos: new Map(conv.turns.map((t, i) => [t.id, i])) }
  const qs = conv.questions.filter((q) => q.evidence.length)
  const qv = await embedMany(qs.map((q) => q.question))
  const needsLlm = chosen.some((m) => m.includes("facet"))
  const tags = needsLlm ? await tagClaims(a.model!, claims) : new Map<string, string[]>()
  const plans = needsLlm ? await mapLimit(qs, 4, (q) => planQuery(a.model!, q.question, conv.speakers)) : qs.map(() => ({ entities: [], facets: [], aggregate: false }))
  qs.forEach((q: Question, qi) => {
    const base: Base = { dense: dense(ix, qv[qi]!), plan: plans[qi]!, tags }
    for (const m of chosen) {
      const r = ranks(q.evidence, METHODS[m]!(ix, qv[qi]!, base))
      const any: Record<string, number | null> = {}
      const all: Record<string, number | null> = {}
      for (const k of KS) { any[k] = r.first !== null && r.first <= k ? 1 : 0; all[k] = r.all !== null && r.all <= k ? 1 : 0 }
      rows[m]!.push({ qid: q.qid, conv: ci, category: q.category, n: q.evidence.length, any, all })
    }
  })
}

function pct(l: Row[], f: (r: Row) => number | null): number | null { return l.length ? +((100 * l.reduce((s, r) => s + (f(r) ?? 0), 0)) / l.length).toFixed(1) : null }
const table: Record<string, Record<string, Record<string, number | null>>> = {}
for (const m of chosen) {
  const l = rows[m]!
  const slices: Record<string, Row[]> = { all: l }
  if (l.some((r) => DEV.includes(r.conv)) && l.some((r) => !DEV.includes(r.conv))) { slices.DEV = l.filter((r) => DEV.includes(r.conv)); slices.TEST = l.filter((r) => !DEV.includes(r.conv)) }
  for (const [c, name] of Object.entries(CATEGORY)) slices[name] = l.filter((r) => r.category === Number(c))
  slices["multi-evidence"] = l.filter((r) => r.n > 1)
  table[m] = Object.fromEntries(Object.entries(slices).map(([s, sl]) => [s, Object.fromEntries([...KS.map((k) => [`any@${k}`, pct(sl, (r) => r.any[k]!)]), ...KS.map((k) => [`all@${k}`, pct(sl, (r) => r.all[k]!)]), ["n", sl.length]])]))
}

const out = { experiment: "E1", ts: new Date().toISOString(), convs, graphOptions: gopts, hippo: HIPPO, graphStats, table }
writeFileSync(join(import.meta.dirname, "..", "results", `e1-${a.label}.json`), JSON.stringify(out, null, 2))
const show = ["all", "multi-hop", "temporal", "open-domain", "single-hop", "multi-evidence"]
console.log(`\nmethod                    | ${show.map((s) => `${s} all@10/all@50`).join(" | ")}`)
for (const m of chosen) console.log(`${m.padEnd(25)} | ${show.map((s) => `${table[m]![s]?.["all@10"] ?? "-"}/${table[m]![s]?.["all@50"] ?? "-"}`.padEnd(s.length + 14)).join(" | ")}`)
