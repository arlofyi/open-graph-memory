// E5: projection cost. Times project() on the real ledgers, then on a
// synthetic ledger scaled to ~100k claims (real claims replicated with
// per-copy entity names, so the graph grows instead of collapsing).
// Synonym linking is O(E^2) in this PoC and is timed separately; the
// rebuild path without it is what an ANN-backed version would approach.
// Pass: < 30 s full rebuild at 100k claims. Kill: > 5 min.
//
//   node bench/e5-cost.ts --target 100000

import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { project, ppr } from "../src/graph.ts"
import { Ledger, type Claim } from "../src/ledger.ts"
import { CACHE } from "../src/llm.ts"
import { loadLocomo, type Turn } from "../src/locomo.ts"

const { values: a } = parseArgs({ options: { target: { type: "string", default: "100000" } } })
const data = loadLocomo()
const realClaims: Claim[] = []
const realTurns: Turn[] = []
for (let ci = 0; ci < 10; ci++) {
  realClaims.push(...new Ledger(join(CACHE, "ledger", `conv${ci}.jsonl`)).live())
  realTurns.push(...data[ci]!.turns.map((t) => ({ ...t, id: `${ci}:${t.id}` })))
}
for (const c of realClaims) c.sources = c.sources.map((s) => `${c.id.split("-")[0]!.slice(1)}:${s}`)
const time = async <T>(f: () => Promise<T> | T) => { const t0 = performance.now(); const v = await f(); return { v, ms: Math.round(performance.now() - t0) } }

const res: Record<string, unknown> = { realClaims: realClaims.length, realTurns: realTurns.length }
res.realNoSynonyms = (await time(() => project(realClaims, realTurns, { synonymThreshold: 1, turnAdjacency: 0 }))).ms
// Synonyms on the real ledger (per conversation, as E1 does): cached embeddings, O(E^2) pairs.
res.realWithSynonymsPerConv = (await time(async () => { for (let ci = 0; ci < 10; ci++) await project(realClaims.filter((c) => c.id.startsWith(`c${ci}-`)), realTurns.filter((t) => t.id.startsWith(`${ci}:`)), { synonymThreshold: 0.92, turnAdjacency: 0 }) })).ms

const copies = Math.ceil(Number(a.target) / realClaims.length)
const synClaims: Claim[] = []
const synTurns: Turn[] = []
for (let k = 0; k < copies; k++) {
  synTurns.push(...realTurns.map((t) => ({ ...t, id: `${k}/${t.id}` })))
  synClaims.push(...realClaims.map((c) => ({ ...c, id: `${k}/${c.id}`, subject: `${c.subject} #${k}`, object: c.object ? `${c.object} #${k}` : null, sources: c.sources.map((s) => `${k}/${s}`) })))
}
const big = await time(() => project(synClaims, synTurns, { synonymThreshold: 1, turnAdjacency: 0 }))
res.synthetic = { claims: synClaims.length, turns: synTurns.length, entities: big.v.stats.entities, edges: big.v.stats.edges, fullRebuildMs: big.ms }
// One PPR query on the 100k graph (seeded on 20 random entities) - the per-question cost.
const seeds = new Map<number, number>()
for (let i = 0; i < 20; i++) seeds.set(big.v.nodes.findIndex((n, j) => n.kind === "entity" && j % 997 === i), 1)
seeds.delete(-1)
res.pprQueryMs = (await time(() => ppr(big.v, seeds, 0.5, 40))).ms
res.heapMB = Math.round(process.memoryUsage().heapUsed / 1e6)
res.pass = { rule: "<30s full rebuild at ~100k claims (no synonym pass)", met: big.ms < 30000 }
console.log(JSON.stringify(res, null, 1))
writeFileSync(join(import.meta.dirname, "..", "results", "e5-cost.json"), JSON.stringify({ experiment: "E5", ts: new Date().toISOString(), ...res }, null, 2))
