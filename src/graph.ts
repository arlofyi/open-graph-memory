// Projection: claim ledger → graph. Disposable and deterministic: the same
// ledger + options always give the same graph, so it is rebuilt, never migrated.
//
// Nodes: entities (normalised mentions) and source turns (passage nodes, as in
// HippoRAG 2). Edges:
//   subject — object          one per claim (the relation)
//   turn — subject/object     the claim's sources (provenance as structure)
//   entity ~ entity           synonym edges above a cosine threshold: mentions
//                             are LINKED, never merged, so a bad link is one
//                             edge, not a corrupted entity
//   turn — next turn          optional, same session

import type { Claim } from "./ledger.ts"
import type { Turn } from "./locomo.ts"
import { dot, embedMany } from "./llm.ts"

export function normEntity(s: string): string {
  return s.toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/^(the|a|an)\s+/, "").replace(/[^\p{L}\p{N}' ]+/gu, " ").replace(/\s+/g, " ").trim()
}

export interface GraphOptions {
  synonymThreshold: number // HippoRAG 2: 0.8
  turnAdjacency: number // weight of next-turn edges; 0 = none
}

export interface Graph {
  n: number
  nodes: Array<{ kind: "entity" | "turn"; key: string }>
  index: Map<string, number> // "e:<norm>" | "t:<turnId>"
  adj: Array<Map<number, number>> // undirected, weighted
  entityTurns: Map<number, Set<number>> // entity node → turn nodes (for specificity)
  claimEntities: Map<string, number[]> // claim id → entity nodes
  synonyms: Array<[number, number, number]> // the linking decisions, kept explicit so they can be audited / retracted
  stats: Record<string, number>
}

function addEdge(g: Graph, a: number, b: number, w: number): void {
  if (a === b) return
  g.adj[a]!.set(b, (g.adj[a]!.get(b) ?? 0) + w)
  g.adj[b]!.set(a, (g.adj[b]!.get(a) ?? 0) + w)
}

function node(g: Graph, kind: "entity" | "turn", key: string): number {
  const k = `${kind === "entity" ? "e" : "t"}:${key}`
  let i = g.index.get(k)
  if (i === undefined) {
    i = g.n++
    g.index.set(k, i)
    g.nodes.push({ kind, key })
    g.adj.push(new Map())
  }
  return i
}

export async function project(claims: Claim[], turns: Turn[], opts: GraphOptions): Promise<Graph> {
  const g: Graph = { n: 0, nodes: [], index: new Map(), adj: [], entityTurns: new Map(), claimEntities: new Map(), synonyms: [], stats: {} }
  for (const t of turns) node(g, "turn", t.id)
  for (const c of claims) {
    const ents = [c.subject, c.object].filter((x): x is string => !!x && normEntity(x).length > 1).map((x) => node(g, "entity", normEntity(x)))
    g.claimEntities.set(c.id, ents)
    if (ents.length === 2) addEdge(g, ents[0]!, ents[1]!, 1)
    for (const s of c.sources) {
      const t = g.index.get(`t:${s}`)
      if (t === undefined) continue
      for (const e of ents) {
        addEdge(g, t, e, 1)
        if (!g.entityTurns.has(e)) g.entityTurns.set(e, new Set())
        g.entityTurns.get(e)!.add(t)
      }
    }
  }
  const entityIdx = g.nodes.map((x, i) => (x.kind === "entity" ? i : -1)).filter((i) => i >= 0)
  let synonyms = 0
  if (opts.synonymThreshold < 1 && entityIdx.length) {
    const vecs = await embedMany(entityIdx.map((i) => g.nodes[i]!.key))
    for (let a = 0; a < vecs.length; a++) for (let b = a + 1; b < vecs.length; b++) {
      const s = dot(vecs[a]!, vecs[b]!)
      if (s >= opts.synonymThreshold) { addEdge(g, entityIdx[a]!, entityIdx[b]!, s); g.synonyms.push([entityIdx[a]!, entityIdx[b]!, s]); synonyms++ }
    }
  }
  if (opts.turnAdjacency > 0) for (let i = 1; i < turns.length; i++) {
    if (turns[i]!.session === turns[i - 1]!.session) addEdge(g, g.index.get(`t:${turns[i - 1]!.id}`)!, g.index.get(`t:${turns[i]!.id}`)!, opts.turnAdjacency)
  }
  g.stats = { entities: entityIdx.length, turns: turns.length, claims: claims.length, synonymEdges: synonyms, edges: g.adj.reduce((s, m) => s + m.size, 0) / 2 }
  return g
}

/** Personalized PageRank by power iteration. `damping` = probability of
 *  following an edge (HippoRAG 2: 0.5); restart goes to the seed distribution. */
export function ppr(g: Graph, seeds: Map<number, number>, damping = 0.5, iters = 40): Float64Array {
  const restart = new Float64Array(g.n)
  let total = 0
  for (const w of seeds.values()) total += w
  if (total <= 0) return restart
  for (const [i, w] of seeds) restart[i] = w / total
  const outW = g.adj.map((m) => { let s = 0; for (const w of m.values()) s += w; return s })
  let p = Float64Array.from(restart)
  for (let it = 0; it < iters; it++) {
    const next = new Float64Array(g.n)
    let dangling = 0
    for (let i = 0; i < g.n; i++) {
      const pi = p[i]!
      if (!pi) continue
      if (outW[i] === 0) { dangling += pi; continue }
      for (const [j, w] of g.adj[i]!) next[j]! += (damping * pi * w) / outW[i]!
    }
    for (let i = 0; i < g.n; i++) next[i]! += (1 - damping + damping * dangling) * restart[i]!
    p = next
  }
  return p
}
