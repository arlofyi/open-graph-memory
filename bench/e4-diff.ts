// E4: does the graph diff between two extractor versions explain itself?
// Replays DEV with ledger A (default: Haiku, .cache/ledger/) and ledger B
// (--tag-b, e.g. Sonnet in .cache/ledger/sonnet/), diffs them, and asks a judge
// to classify a seeded sample of the diff:
//   gained claim  → true gain (supported by its turns) | false gain
//   lost claim    → true loss (it was unsupported)     | real loss (supported, now missing)
// "Explainable" = the judge can place the item in one of these with the
// evidence shown. Pass: >= 90% of the top-50 sampled changes explainable.
//
//   node bench/e4-diff.ts --tag-b sonnet --convs 0,1

import { writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { diffLedgers } from "../src/diff.ts"
import { Ledger, type Claim } from "../src/ledger.ts"
import { CACHE, chat, mapLimit, parseJson } from "../src/llm.ts"
import { loadLocomo } from "../src/locomo.ts"

const { values: a } = parseArgs({ options: { convs: { type: "string", default: "0,1" }, "tag-a": { type: "string", default: "" }, "tag-b": { type: "string", default: "sonnet" }, judge: { type: "string", default: "claude-sonnet-5-5" }, n: { type: "string", default: "50" } } })
const data = loadLocomo()
const convs = a.convs!.split(",").map(Number)
const per: Record<string, unknown>[] = []
const items: Array<{ side: "gained" | "lost"; conv: number; c: Claim }> = []
for (const ci of convs) {
  const A = new Ledger(join(CACHE, "ledger", a["tag-a"]!, `conv${ci}.jsonl`)).live()
  const B = new Ledger(join(CACHE, "ledger", a["tag-b"]!, `conv${ci}.jsonl`)).live()
  const d = await diffLedgers(A, B, data[ci]!.turns.map((t) => t.id))
  per.push({ conv: ci, claimsA: A.length, claimsB: B.length, matched: d.matched.length, lost: d.onlyA.length, gained: d.onlyB.length, whenChanged: d.matched.filter((m) => m.whenChanged).length, entitiesOnlyA: d.entities.onlyA.length, entitiesOnlyB: d.entities.onlyB.length, entitiesBoth: d.entities.both, coverage: d.coverage })
  items.push(...d.onlyB.map((c) => ({ side: "gained" as const, conv: ci, c })), ...d.onlyA.map((c) => ({ side: "lost" as const, conv: ci, c })))
  console.log(JSON.stringify(per.at(-1)))
}
// Seeded interleaved sample: half gained, half lost.
const pick = (side: string, n: number) => items.filter((x) => x.side === side).filter((_, i, l) => i % Math.max(1, Math.floor(l.length / n)) === 0).slice(0, n)
const sample = [...pick("gained", Math.ceil(Number(a.n) / 2)), ...pick("lost", Math.floor(Number(a.n) / 2))]
const SYS = `You audit one claim that an extractor produced from the cited chat turns. JSON: {"supported":"yes"|"partial"|"no","explainable":true|false,"why":""}
explainable = you can tell from the turns shown whether the claim is right or wrong.`
const verdicts = await mapLimit(sample, 3, async ({ side, conv, c }) => {
  const turns = data[conv]!.turns.filter((t) => c.sources.includes(t.id)).map((t) => `[${t.id}] (${t.date.toISOString().slice(0, 10)}) ${t.text}`).join("\n")
  const v = parseJson<{ supported: string; explainable: boolean; why: string }>(await chat(a.judge!, [{ role: "system", content: SYS }, { role: "user", content: `Turns:\n${turns}\n\nClaim: ${c.text}` }], { json: true, maxTokens: 300 }))
  return { side, text: c.text, ...v }
})
const f = (l: typeof verdicts, p: (v: (typeof verdicts)[number]) => boolean) => (l.length ? +((100 * l.filter(p).length) / l.length).toFixed(1) : null)
const g = verdicts.filter((v) => v.side === "gained"), l = verdicts.filter((v) => v.side === "lost")
const summary = {
  explainable: f(verdicts, (v) => v.explainable),
  gained: { n: g.length, trueGain: f(g, (v) => v.supported === "yes"), falseGain: f(g, (v) => v.supported === "no") },
  lost: { n: l.length, realLoss: f(l, (v) => v.supported === "yes"), trueLossOfBadClaim: f(l, (v) => v.supported === "no") },
}
console.log(JSON.stringify(summary))
writeFileSync(join(import.meta.dirname, "..", "results", "e4-diff.json"), JSON.stringify({ experiment: "E4", ts: new Date().toISOString(), a: a["tag-a"] || "haiku", b: a["tag-b"], per, summary, pass: { rule: ">=90% explainable", met: (summary.explainable ?? 0) >= 90 } }, null, 2))
