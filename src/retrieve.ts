// Retrieval methods. Every method returns a ranked list of TURN ids, so all of
// them are scored on the same unit (k counts turns).
//
//   dense        turn embedded with the previous turn (the baseline)
//   dense+next   dense, each hit followed by its next turn (cheap baseline from
//                the literature: one-hop next-turn expansion)
//   claims       question vs claim embeddings; emit each claim's source turns
//   ppr          HippoRAG-2-style Personalized PageRank over entity+turn graph
//   rrf(a,b,…)   reciprocal rank fusion, k = 60

import type { Claim } from "./ledger.ts"
import type { Turn } from "./locomo.ts"
import { dot } from "./llm.ts"
import { normEntity, ppr, type Graph } from "./graph.ts"

export interface Index {
  turns: Turn[]
  turnVec: Float32Array[] // dense key per turn
  turnVecX?: Float32Array[] // turn + its claims ("document expansion" key)
  claims: Claim[]
  claimVec: Float32Array[]
  graph: Graph
  pos: Map<string, number> // turn id → position in `turns`
}

export type Ranking = string[]

function topBy<T>(items: T[], score: (x: T) => number, k: number): Array<{ x: T; s: number }> {
  return items.map((x) => ({ x, s: score(x) })).sort((a, b) => b.s - a.s).slice(0, k)
}

export function dense(ix: Index, q: Float32Array): Ranking {
  return ix.turns.map((t, i) => ({ id: t.id, s: dot(q, ix.turnVec[i]!) })).sort((a, b) => b.s - a.s).map((x) => x.id)
}

export function denseX(ix: Index, q: Float32Array): Ranking {
  const v = ix.turnVecX ?? ix.turnVec
  return ix.turns.map((t, i) => ({ id: t.id, s: dot(q, v[i]!) })).sort((a, b) => b.s - a.s).map((x) => x.id)
}

export function withNext(ix: Index, r: Ranking): Ranking {
  const out: string[] = []
  const seen = new Set<string>()
  const push = (id: string | undefined) => { if (id && !seen.has(id)) { seen.add(id); out.push(id) } }
  for (const id of r) {
    push(id)
    const p = ix.pos.get(id)!
    const nx = ix.turns[p + 1]
    if (nx && nx.session === ix.turns[p]!.session) push(nx.id)
  }
  return out
}

export function claimsRank(ix: Index, q: Float32Array, fallback: Ranking): Ranking {
  const out: string[] = []
  const seen = new Set<string>()
  for (const { x } of topBy(ix.claims.map((c, i) => i), (i) => dot(q, ix.claimVec[i]!), ix.claims.length)) {
    for (const s of ix.claims[x]!.sources) if (!seen.has(s)) { seen.add(s); out.push(s) }
  }
  for (const id of fallback) if (!seen.has(id)) { seen.add(id); out.push(id) }
  return out
}

export interface PprOptions {
  topClaims: number // query→claim links that seed entities (HippoRAG 2: 5)
  passageWeight: number // dense seed weight on turn nodes (HippoRAG 2: 0.05)
  specificity: boolean // divide entity seed by |turns it appears in| (HippoRAG v1)
  damping: number // HippoRAG 2: 0.5
}

export function pprRank(ix: Index, q: Float32Array, o: PprOptions): Ranking {
  const g = ix.graph
  const seeds = new Map<number, number>()
  // Entity seeds: mean score of the top linked claims that mention them.
  const top = topBy(ix.claims.map((_, i) => i), (i) => dot(q, ix.claimVec[i]!), o.topClaims)
  const acc = new Map<number, { s: number; n: number }>()
  for (const { x, s } of top) for (const e of g.claimEntities.get(ix.claims[x]!.id) ?? []) {
    const a = acc.get(e) ?? { s: 0, n: 0 }
    a.s += s; a.n++
    acc.set(e, a)
  }
  // Normalise the two seed families separately, then mix by passageWeight,
  // so the turn seeds' share is a real fraction regardless of graph size.
  let eTot = 0
  const eSeed = new Map<number, number>()
  for (const [e, a] of acc) {
    const spec = o.specificity ? 1 / Math.max(1, g.entityTurns.get(e)?.size ?? 1) : 1
    const w = (a.s / a.n) * spec
    eSeed.set(e, w); eTot += w
  }
  const tScores = ix.turns.map((_, i) => Math.max(0, dot(q, ix.turnVec[i]!)))
  const tTot = tScores.reduce((s, x) => s + x, 0)
  for (const [e, w] of eSeed) seeds.set(e, ((1 - o.passageWeight) * w) / (eTot || 1))
  ix.turns.forEach((t, i) => {
    const n = g.index.get(`t:${t.id}`)!
    seeds.set(n, (seeds.get(n) ?? 0) + (o.passageWeight * tScores[i]!) / (tTot || 1))
  })
  const p = ppr(g, seeds, o.damping)
  return ix.turns.map((t) => ({ id: t.id, s: p[g.index.get(`t:${t.id}`)!]! })).sort((a, b) => b.s - a.s).map((x) => x.id)
}

export function rrf(rankings: Ranking[], k = 60, weights?: number[]): Ranking {
  const s = new Map<string, number>()
  rankings.forEach((r, ri) => r.forEach((id, i) => s.set(id, (s.get(id) ?? 0) + (weights?.[ri] ?? 1) / (k + i + 1))))
  return [...s.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id)
}

/** Typed traversal: claims ABOUT the planned entities (subject or object
 *  mention contains the entity) whose facets intersect the plan's, ranked by
 *  similarity; their source turns first, then `fallback`. With no entity
 *  match it degrades to a facet-only filter. */
export function facetRank(ix: Index, q: Float32Array, plan: { entities: string[]; facets: string[] }, tags: Map<string, string[]>, fallback: Ranking, mmr = 0): Ranking {
  const ents = plan.entities.map((e) => normEntity(e)).filter((e) => e.length > 1)
  const about = (c: Claim) => {
    const s = normEntity(c.subject)
    const o = c.object ? normEntity(c.object) : ""
    return ents.some((e) => s.includes(e) || o.includes(e) || (e.includes(s) && s.length > 2))
  }
  const facet = (c: Claim) => !plan.facets.length || (tags.get(c.id) ?? []).some((f) => plan.facets.includes(f))
  let cand = ix.claims.map((c, i) => i).filter((i) => about(ix.claims[i]!) && facet(ix.claims[i]!))
  if (!cand.length) cand = ix.claims.map((c, i) => i).filter((i) => facet(ix.claims[i]!))
  const out: string[] = []
  const seen = new Set<string>()
  const order = mmr > 0 ? mmrOrder(cand, (i) => dot(q, ix.claimVec[i]!), (i, j) => dot(ix.claimVec[i]!, ix.claimVec[j]!), mmr) : cand.sort((x, y) => dot(q, ix.claimVec[y]!) - dot(q, ix.claimVec[x]!))
  for (const i of order) {
    for (const s of ix.claims[i]!.sources) if (!seen.has(s)) { seen.add(s); out.push(s) }
  }
  for (const id of fallback) if (!seen.has(id)) { seen.add(id); out.push(id) }
  return out
}

/** Maximal marginal relevance: lambda * relevance - (1 - lambda) * max
 *  similarity to anything already picked. Aggregation questions need breadth
 *  (pottery AND camping AND painting), which pure similarity ranking collapses. */
export function mmrOrder(cand: number[], rel: (i: number) => number, sim: (i: number, j: number) => number, lambda: number): number[] {
  const left = new Set(cand)
  const out: number[] = []
  const maxSim = new Map<number, number>(cand.map((i) => [i, -Infinity]))
  const r = new Map(cand.map((i) => [i, rel(i)]))
  while (left.size) {
    let best = -1, bestS = -Infinity
    for (const i of left) {
      const s = lambda * r.get(i)! - (1 - lambda) * (out.length ? maxSim.get(i)! : 0)
      if (s > bestS) { bestS = s; best = i }
    }
    left.delete(best); out.push(best)
    for (const i of left) maxSim.set(i, Math.max(maxSim.get(i)!, sim(i, best)))
  }
  return out
}
