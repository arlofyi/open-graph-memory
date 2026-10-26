// Failure inspection: for DEV questions of one category, where does each
// evidence turn land under a method, and is it cited by any claim?
//   node bench/inspect.ts --conv 0 --category 1 --n 12
import { join } from "node:path"
import { parseArgs } from "node:util"
import { project } from "../src/graph.ts"
import { Ledger } from "../src/ledger.ts"
import { CACHE, embedMany } from "../src/llm.ts"
import { loadLocomo } from "../src/locomo.ts"
import { claimsRank, dense, rrf, type Index } from "../src/retrieve.ts"
const { values: a } = parseArgs({ options: { conv: { type: "string", default: "0" }, category: { type: "string", default: "1" }, n: { type: "string", default: "12" } } })
const conv = loadLocomo()[Number(a.conv)]!
const claims = new Ledger(join(CACHE, "ledger", `conv${a.conv}.jsonl`)).live()
const turnVec = await embedMany(conv.turns.map((t) => (t.prev ? `${t.prev}\n${t.text}` : t.text)))
const claimVec = await embedMany(claims.map((c) => `${c.text}${c.validFrom ? ` (${c.validFrom}${c.validTo ? ` to ${c.validTo}` : ""})` : ""}`))
const ix: Index = { turns: conv.turns, turnVec, claims, claimVec, graph: await project(claims, conv.turns, { synonymThreshold: 1, turnAdjacency: 0 }), pos: new Map(conv.turns.map((t, i) => [t.id, i])) }
const cited = new Map<string, string[]>()
for (const c of claims) for (const s of c.sources) cited.set(s, [...(cited.get(s) ?? []), c.text])
const qs = conv.questions.filter((q) => q.category === Number(a.category) && q.evidence.length)
const qv = await embedMany(qs.map((q) => q.question))
qs.slice(0, Number(a.n)).forEach((q, i) => {
  const d = dense(ix, qv[i]!)
  const r = rrf([d, claimsRank(ix, qv[i]!, d)])
  console.log(`\nQ ${q.qid}: ${q.question}  →  ${q.answer}`)
  for (const e of q.evidence) {
    const t = conv.turns[ix.pos.get(e)!]!
    console.log(`  ${e} rank=${r.indexOf(e) + 1} dense=${d.indexOf(e) + 1} claims=${(cited.get(e) ?? []).length} | ${t.text.slice(0, 140)}`)
    for (const c of (cited.get(e) ?? []).slice(0, 2)) console.log(`      · ${c.slice(0, 130)}`)
  }
})
