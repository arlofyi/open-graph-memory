// E7 report: reads the runner's judged per-question files for every system in
// .cache/h2h/<split>/predicted_<name>/ and writes results/e7-h2h-<split>.json.
//
// Per system and cutoff: judged accuracy (overall and per category), the
// context the answerer was given (≈ tokens, chars / 4, of the top-k memories),
// and, where a system returns LoCoMo turn ids (open-graph-memory), evidence
// recall all@k. Paired McNemar (exact, two-sided) of a reference system against
// every other, on the questions both judged.
//
//   node bench/h2h/report.ts --split dev [--ref ogm-r2a]

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"

const { values: a } = parseArgs({ options: { split: { type: "string", default: "dev" }, ref: { type: "string", default: "ogm-r2a" } } })
const ROOT = join(import.meta.dirname, "..", "..")
const DIR = join(ROOT, ".cache", "h2h", a.split!)
const CUTS = ["top_10", "top_50", "top_200"]
const CATS = ["multi-hop", "temporal", "open-domain", "single-hop"]

interface Q {
  question_id: string
  category_name: string
  evidence: string[]
  retrieval: { search_results: Array<{ id: string; memory: string }> }
  cutoff_results?: Record<string, { judgment: string; memories_evaluated: number }>
}

const systems = existsSync(DIR) ? readdirSync(DIR).filter((d) => d.startsWith("predicted_")).map((d) => d.slice("predicted_".length)).filter((s) => !s.startsWith("smoke")) : []
const data = new Map<string, Map<string, Q>>()
for (const s of systems) {
  const m = new Map<string, Q>()
  for (const f of readdirSync(join(DIR, `predicted_${s}`)).filter((f) => /^conv\d+_q\d+\.json$/.test(f))) {
    const q = JSON.parse(readFileSync(join(DIR, `predicted_${s}`, f), "utf8")) as Q
    if (q.cutoff_results) m.set(q.question_id, q)
  }
  data.set(s, m)
}

const ok = (q: Q, cut: string) => q.cutoff_results?.[cut]?.judgment === "CORRECT"
const pct = (n: number, d: number) => (d ? +((100 * n) / d).toFixed(1) : null)
const k = (cut: string) => Number(cut.slice(4))

/** Exact two-sided McNemar on discordant pairs. */
function mcnemar(b: number, c: number): number {
  const n = b + c
  if (!n) return 1
  const lo = Math.min(b, c)
  let p = 0
  let comb = 1
  for (let i = 0; i <= n; i++) {
    if (i > 0) comb = (comb * (n - i + 1)) / i
    if (i <= lo) p += comb
  }
  return Math.min(1, (2 * p) / 2 ** n)
}

const table: Record<string, unknown> = {}
for (const s of systems) {
  const qs = [...data.get(s)!.values()]
  const row: Record<string, unknown> = { n: qs.length }
  for (const cut of CUTS) {
    const top = (q: Q) => q.retrieval.search_results.slice(0, k(cut))
    const tokens = qs.map((q) => top(q).reduce((t, r) => t + r.memory.length, 0) / 4)
    const turnIds = qs.every((q) => top(q).every((r) => /^D\d+:\d+$/.test(r.id)))
    const withEv = qs.filter((q) => q.evidence?.length)
    row[cut] = {
      accuracy: pct(qs.filter((q) => ok(q, cut)).length, qs.length),
      ...Object.fromEntries(CATS.map((c) => { const l = qs.filter((q) => q.category_name === c); return [c, pct(l.filter((q) => ok(q, cut)).length, l.length)] })),
      context_tokens_mean: Math.round(tokens.reduce((x, y) => x + y, 0) / (tokens.length || 1)),
      ...(turnIds ? { evidence_all: pct(withEv.filter((q) => q.evidence.every((e) => top(q).some((r) => r.id === e))).length, withEv.length) } : {}),
    }
  }
  table[s] = row
}

const paired: Record<string, unknown> = {}
const ref = data.get(a.ref!)
if (ref) {
  for (const s of systems.filter((x) => x !== a.ref)) {
    const other = data.get(s)!
    const ids = [...ref.keys()].filter((id) => other.has(id))
    paired[s] = Object.fromEntries(CUTS.map((cut) => {
      let win = 0
      let loss = 0
      for (const id of ids) {
        const r = ok(ref.get(id)!, cut)
        const o = ok(other.get(id)!, cut)
        if (r && !o) win++
        if (o && !r) loss++
      }
      return [cut, { n: ids.length, [`${a.ref}_wins`]: win, [`${s}_wins`]: loss, p: +mcnemar(win, loss).toPrecision(3) }]
    }))
  }
}

const out = { experiment: "E7 head-to-head (Mem0's LoCoMo harness, unmodified)", split: a.split, ts: new Date().toISOString(), answerer: "claude-haiku-5-5", judge: "claude-haiku-5-5", table, paired_vs: a.ref, paired }
writeFileSync(join(ROOT, "results", `e7-h2h-${a.split}.json`), JSON.stringify(out, null, 2))

const fmt = (v: unknown) => (v == null ? "n/a" : String(v))
console.log(`| system | n | acc@10 | acc@50 | tokens@10 | tokens@50 | ${CATS.map((c) => `${c}@10`).join(" | ")} | evidence all@10 |`)
console.log(`|---|---|---|---|---|---|${CATS.map(() => "---").join("|")}|---|`)
for (const s of systems) {
  const r = table[s] as Record<string, Record<string, unknown>> & { n: number }
  console.log(`| ${s} | ${r.n} | ${fmt(r.top_10!.accuracy)} | ${fmt(r.top_50!.accuracy)} | ${fmt(r.top_10!.context_tokens_mean)} | ${fmt(r.top_50!.context_tokens_mean)} | ${CATS.map((c) => fmt(r.top_10![c])).join(" | ")} | ${fmt(r.top_10!.evidence_all)} |`)
}
console.log(JSON.stringify(paired, null, 1))
