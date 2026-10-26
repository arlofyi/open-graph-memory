// Graph diff: what changed between two projections of the SAME sources
// (e.g. one ledger extracted by model A, replayed by model B). Claims are
// aligned semantically (greedy, cosine >= threshold), because two extractors
// never phrase a claim identically; entities are compared by normalised key.

import type { Claim } from "./ledger.ts"
import { normEntity } from "./graph.ts"
import { dot, embedMany } from "./llm.ts"

export interface ClaimDiff {
  matched: Array<{ a: Claim; b: Claim; sim: number; whenChanged: boolean; sourcesChanged: boolean }>
  onlyA: Claim[] // lost on replay
  onlyB: Claim[] // gained on replay
  entities: { onlyA: string[]; onlyB: string[]; both: number }
  coverage: { a: number; b: number } // share of source turns cited by >= 1 claim
}

export async function diffLedgers(A: Claim[], B: Claim[], turnIds: string[], threshold = 0.9): Promise<ClaimDiff> {
  const va = await embedMany(A.map((c) => c.text))
  const vb = await embedMany(B.map((c) => c.text))
  const pairs: Array<{ i: number; j: number; s: number }> = []
  for (let i = 0; i < A.length; i++) for (let j = 0; j < B.length; j++) {
    // Only compare claims that share a source turn: alignment is about the same evidence.
    if (!A[i]!.sources.some((s) => B[j]!.sources.includes(s))) continue
    const s = dot(va[i]!, vb[j]!)
    if (s >= threshold) pairs.push({ i, j, s })
  }
  pairs.sort((x, y) => y.s - x.s)
  const ua = new Set<number>(), ub = new Set<number>()
  const matched: ClaimDiff["matched"] = []
  for (const p of pairs) {
    if (ua.has(p.i) || ub.has(p.j)) continue
    ua.add(p.i); ub.add(p.j)
    const a = A[p.i]!, b = B[p.j]!
    matched.push({ a, b, sim: p.s, whenChanged: (a.validFrom ?? "") !== (b.validFrom ?? ""), sourcesChanged: a.sources.slice().sort().join() !== b.sources.slice().sort().join() })
  }
  const ents = (cs: Claim[]) => new Set(cs.flatMap((c) => [c.subject, c.object]).filter((x): x is string => !!x).map(normEntity).filter((x) => x.length > 1))
  const ea = ents(A), eb = ents(B)
  const cov = (cs: Claim[]) => new Set(cs.flatMap((c) => c.sources)).size / turnIds.length
  return {
    matched,
    onlyA: A.filter((_, i) => !ua.has(i)),
    onlyB: B.filter((_, j) => !ub.has(j)),
    entities: { onlyA: [...ea].filter((e) => !eb.has(e)), onlyB: [...eb].filter((e) => !ea.has(e)), both: [...ea].filter((e) => eb.has(e)).length },
    coverage: { a: +cov(A).toFixed(3), b: +cov(B).toFixed(3) },
  }
}
