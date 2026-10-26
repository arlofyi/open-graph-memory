// E3: is entity resolution conservative enough?
// The projection LINKS mentions above a cosine threshold; it never merges.
// Here we (1) judge a sample of those links (same real-world entity?) and
// (2) show what a naive MERGE (transitive closure of the same links) would
// have done: the cluster-size tail is the blast radius links avoid.
// Pass: < 5 wrong links per 1k entities. Kill: > 20.
//
//   node bench/e3-resolution.ts --convs 0,1,2,3,4,5,6,7,8,9 --threshold 0.92

import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { project } from "../src/graph.ts"
import { Ledger } from "../src/ledger.ts"
import { CACHE, chat, mapLimit, parseJson } from "../src/llm.ts"
import { loadLocomo } from "../src/locomo.ts"

const { values: a } = parseArgs({ options: { convs: { type: "string", default: "0,1" }, threshold: { type: "string", default: "0.92" }, judge: { type: "string", default: "claude-sonnet-5-5" }, n: { type: "string", default: "120" } } })
const data = loadLocomo()
const links: Array<{ conv: number; x: string; y: string; w: number }> = []
let entities = 0
const clusterTail: number[] = []
for (const ci of a.convs!.split(",").map(Number)) {
  const g = await project(new Ledger(join(CACHE, "ledger", `conv${ci}.jsonl`)).live(), data[ci]!.turns, { synonymThreshold: Number(a.threshold), turnAdjacency: 0 })
  entities += g.stats.entities!
  // union-find over synonym edges = what a merge would produce
  const parent = g.nodes.map((_, i) => i)
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i]!)))
  for (const [i, j, w] of g.synonyms) {
    links.push({ conv: ci, x: g.nodes[i]!.key, y: g.nodes[j]!.key, w })
    parent[find(i)] = find(j)
  }
  const sizes = new Map<number, number>()
  g.nodes.forEach((n, i) => { if (n.kind === "entity") sizes.set(find(i), (sizes.get(find(i)) ?? 0) + 1) })
  clusterTail.push(...[...sizes.values()].filter((s) => s > 1))
}
const step = Math.max(1, Math.floor(links.length / Number(a.n)))
const sample = links.filter((_, i) => i % step === 0).slice(0, Number(a.n))
const SYS = `Two mention strings were extracted from the same conversation. Do they refer to the SAME real-world entity (person, pet, place, object, organisation, event or concept)? Different things that are merely related (a person and their pet, "pottery" and "pottery class", "mom" and "Caroline's mom" when unclear) are NOT the same. JSON {"same":"yes"|"no"|"unclear","why":""}`
const v = await mapLimit(sample, 3, async (l) => ({ ...l, ...parseJson<{ same: string; why: string }>(await chat(a.judge!, [{ role: "system", content: SYS }, { role: "user", content: `A: "${l.x}"\nB: "${l.y}"` }], { json: true, maxTokens: 200 })) }))
const wrongRate = v.filter((x) => x.same === "no").length / Math.max(1, v.length)
const wrongPer1k = +((1000 * wrongRate * links.length) / entities).toFixed(1)
const out = {
  experiment: "E3", ts: new Date().toISOString(), threshold: Number(a.threshold), entities, links: links.length, judged: v.length,
  linkPrecision: +(100 * v.filter((x) => x.same === "yes").length / Math.max(1, v.length)).toFixed(1),
  wrongLinksPer1kEntities: wrongPer1k,
  mergeWouldCreate: { clusters: clusterTail.length, largest: Math.max(0, ...clusterTail), over3: clusterTail.filter((s) => s > 3).length },
  wrong: v.filter((x) => x.same === "no").map((x) => `${x.x} ~ ${x.y} (${x.w.toFixed(3)})`),
  pass: { rule: "<5 wrong links / 1k entities", met: wrongPer1k < 5 }, kill: { rule: ">20", met: wrongPer1k > 20 },
}
console.log(JSON.stringify(out, null, 1))
writeFileSync(join(import.meta.dirname, "..", "results", `e3-t${a.threshold}.json`), JSON.stringify(out, null, 2))
