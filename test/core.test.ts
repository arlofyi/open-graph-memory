import assert from "node:assert/strict"
import { test } from "node:test"
import { normEntity, ppr, type Graph } from "../src/graph.ts"
import { Ledger, type Claim } from "../src/ledger.ts"
import { rrf } from "../src/retrieve.ts"

const claim = (id: string, over: Partial<Claim> = {}): Claim => ({
  id, kind: "assert", subject: "Ana", predicate: "likes", object: "tea", qualifiers: {}, text: "Ana likes tea.",
  validFrom: null, validTo: null, recordedAt: "2026-01-01T00:00:00Z", sources: ["D1:1"], confidence: 1, actor: "ai", extractor: "test", ...over,
})

test("ledger refuses a claim with no source", () => {
  assert.throws(() => new Ledger().append(claim("c1", { sources: [] })), /no source/)
})

test("retraction applies from its recorded time, not before", () => {
  const l = new Ledger()
  l.append(claim("c1"))
  l.append(claim("r1", { kind: "retract", retracts: "c1", recordedAt: "2026-03-01T00:00:00Z" }))
  assert.equal(l.live("2026-02-01T00:00:00Z").length, 1)
  assert.equal(l.live().length, 0)
})

test("same_as / not_same_as / retract never count as live assertions", () => {
  const l = new Ledger()
  l.append(claim("s1", { kind: "same_as" }))
  l.append(claim("n1", { kind: "not_same_as" }))
  assert.equal(l.live().length, 0)
})

function line(n: number): Graph {
  const g: Graph = { n, nodes: Array.from({ length: n }, (_, i) => ({ kind: "entity" as const, key: String(i) })), index: new Map(), adj: Array.from({ length: n }, () => new Map()), entityTurns: new Map(), claimEntities: new Map(), synonyms: [], stats: {} }
  for (let i = 0; i + 1 < n; i++) { g.adj[i]!.set(i + 1, 1); g.adj[i + 1]!.set(i, 1) }
  return g
}

test("PPR conserves probability mass, including a dangling node", () => {
  const g = line(5)
  g.n = 6; g.nodes.push({ kind: "entity", key: "lonely" }); g.adj.push(new Map())
  const p = ppr(g, new Map([[0, 1], [5, 1]]), 0.5, 60)
  assert.ok(Math.abs(p.reduce((s, x) => s + x, 0) - 1) < 1e-9)
})

test("PPR decays with distance from the seed", () => {
  const p = ppr(line(5), new Map([[0, 1]]), 0.5, 60)
  assert.ok(p[0]! > p[1]! && p[1]! > p[2]! && p[2]! > p[3]! && p[3]! > p[4]!)
})

test("RRF rewards agreement between rankings", () => {
  assert.deepEqual(rrf([["a", "b", "c"], ["b", "a", "d"]]).slice(0, 2).sort(), ["a", "b"])
  assert.equal(rrf([["x", "b"], ["y", "b"]])[0], "b")
})

test("entity normalisation folds case, articles and accents, keeps possessives", () => {
  assert.equal(normEntity("The Pottery Class"), "pottery class")
  assert.equal(normEntity("Zoë"), "zoe")
  assert.equal(normEntity("Caroline's mom"), "caroline's mom")
})
