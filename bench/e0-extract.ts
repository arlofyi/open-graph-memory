// E0: can we pull claims out of CONVERSATION at usable precision?
// (Fact extractors built for documents yield few or no facts from chat.)
//
//   node bench/e0-extract.ts --convs 0,1 --model claude-haiku-5-5 --judge claude-sonnet-5-5 --judge-n 100
//
// Writes the ledger to .cache/ledger/conv<N>.jsonl (gitignored: derived from
// CC BY-NC data) and the metrics to results/e0-<label>.json.

import { mkdirSync, rmSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { parseArgs } from "node:util"
import { extractSession, EXTRACTOR_VERSION } from "../src/extract.ts"
import { Ledger } from "../src/ledger.ts"
import { CACHE, chat, mapLimit, parseJson } from "../src/llm.ts"
import { CATEGORY, loadLocomo } from "../src/locomo.ts"

const { values: a } = parseArgs({ options: {
  convs: { type: "string", default: "0,1" },
  model: { type: "string", default: "claude-haiku-5-5" },
  judge: { type: "string", default: "claude-sonnet-5-5" },
  "judge-n": { type: "string", default: "100" },
  concurrency: { type: "string", default: "3" },
  label: { type: "string", default: "dev" },
  tag: { type: "string", default: "" }, // ledger variant, e.g. "sonnet" -> .cache/ledger/sonnet/
} })
const convs = a.convs!.split(",").map(Number)
const data = loadLocomo()
const log = (m: string) => console.log(`[e0 ${new Date().toISOString().slice(11, 19)}] ${m}`)

// Deterministic sample (mulberry32) so the judged set is reproducible.
function rng(seed: number) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296 } }

const perConv: Record<string, unknown>[] = []
const all: Array<{ conv: number; claim: import("../src/ledger.ts").Claim }> = []
const t0 = Date.now()
for (const ci of convs) {
  const conv = data[ci]!
  const dir = join(CACHE, "ledger", a.tag!)
  mkdirSync(dir, { recursive: true })
  const path = join(dir, `conv${ci}.jsonl`)
  rmSync(path, { force: true }) // the ledger is re-derived from cached extractions; never hand-edited
  const ledger = new Ledger(path)
  const res = await mapLimit(conv.sessions, Number(a.concurrency), (s) => extractSession(a.model!, ci, s.key, s.date, s.rawDate, s.turns))
  for (const cs of res) for (const c of cs) { ledger.append(c); all.push({ conv: ci, claim: c }) }
  const cited = new Set(ledger.claims.flatMap((c) => c.sources))
  const ev = conv.questions.filter((q) => q.evidence.length)
  const evTurns = new Set(ev.flatMap((q) => q.evidence))
  const qAll = ev.filter((q) => q.evidence.every((e) => cited.has(e)))
  const byCat = Object.fromEntries(Object.entries(CATEGORY).map(([k, name]) => {
    const l = ev.filter((q) => q.category === Number(k))
    return [name, l.length ? +(l.filter((q) => q.evidence.every((e) => cited.has(e))).length / l.length).toFixed(3) : null]
  }))
  const row = {
    conv: ci,
    sessions: conv.sessions.length,
    turns: conv.turns.length,
    claims: ledger.claims.length,
    claimsPerTurn: +(ledger.claims.length / conv.turns.length).toFixed(2),
    turnCoverage: +(cited.size / conv.turns.length).toFixed(3),
    evidenceTurnCoverage: +([...evTurns].filter((t) => cited.has(t)).length / evTurns.size).toFixed(3),
    questionsAllEvidenceCited: +(qAll.length / ev.length).toFixed(3),
    questionsAllEvidenceCitedByCategory: byCat,
    withWhen: +(ledger.claims.filter((c) => c.validFrom).length / ledger.claims.length).toFixed(3),
  }
  perConv.push(row)
  log(JSON.stringify(row))
}

// Precision: an LLM judge (NOT a human) checks a seeded random sample against
// the cited turns. Reported as such; a human spot-check is noted separately.
const n = Math.min(Number(a["judge-n"]), all.length)
const r = rng(42)
const sample = [...all].map((x) => ({ x, k: r() })).sort((p, q) => p.k - q.k).slice(0, n).map((p) => p.x)
const JUDGE = `You audit claims extracted from a chat transcript. Given the cited turns (with the session date) and one claim, answer in JSON:
{"supported":"yes"|"partial"|"no","standalone":true|false,"when_ok":"yes"|"no"|"n/a","why":""}
- supported: yes = every part of the claim (including qualifiers) is stated or directly implied by the cited turns; partial = the core is right but a detail is wrong or invented; no = not supported.
- standalone: the text is understandable alone (no unresolved pronouns, full names).
- when_ok: if the claim has a date, is it a correct resolution against the session date? n/a if no date.`
const verdicts = await mapLimit(sample, 3, async ({ conv, claim }) => {
  const turns = data[conv]!.turns.filter((t) => claim.sources.includes(t.id))
  const ctx = turns.map((t) => `[${t.id}] (${t.date.toISOString().slice(0, 10)}) ${t.text}`).join("\n")
  const msg = `Cited turns:\n${ctx}\n\nClaim: ${claim.text}\nQualifiers: ${JSON.stringify(claim.qualifiers)}\nDate: ${claim.validFrom ?? "none"}${claim.validTo ? ` to ${claim.validTo}` : ""}`
  const v = parseJson<{ supported: string; standalone: boolean; when_ok: string; why: string }>(await chat(a.judge!, [{ role: "system", content: JUDGE }, { role: "user", content: msg }], { json: true, maxTokens: 400 }))
  return { conv, id: claim.id, text: claim.text, ...v }
})
const frac = (f: (v: (typeof verdicts)[number]) => boolean, l = verdicts) => (l.length ? +(l.filter(f).length / l.length).toFixed(3) : null)
const dated = verdicts.filter((v) => v.when_ok !== "n/a")
const precision = {
  judge: a.judge,
  n: verdicts.length,
  supported: frac((v) => v.supported === "yes"),
  supportedOrPartial: frac((v) => v.supported !== "no"),
  standalone: frac((v) => v.standalone === true),
  dateCorrect: frac((v) => v.when_ok === "yes", dated),
  datedN: dated.length,
}
log(`precision ${JSON.stringify(precision)}`)

const totals = { turns: 0, claims: 0 }
for (const r of perConv as Array<{ turns: number; claims: number }>) { totals.turns += r.turns; totals.claims += r.claims }
const result = {
  experiment: "E0",
  ts: new Date().toISOString(),
  extractor: `${EXTRACTOR_VERSION}/${a.model}`,
  convs,
  claimsPerTurn: +(totals.claims / totals.turns).toFixed(2),
  perConv,
  precision,
  failures: verdicts.filter((v) => v.supported !== "yes").map((v) => ({ id: v.id, supported: v.supported })), // no rationale: it quotes the CC BY-NC data
  pass: { rule: "≥0.5 claims/turn at ≥85% supported", met: totals.claims / totals.turns >= 0.5 && (precision.supported ?? 0) >= 0.85 },
  kill: { rule: "<70% supported", met: (precision.supported ?? 1) < 0.7 },
  wallSeconds: Math.round((Date.now() - t0) / 1000),
}
mkdirSync(join(import.meta.dirname, "..", "results"), { recursive: true })
writeFileSync(join(import.meta.dirname, "..", "results", `e0-${a.label}.json`), JSON.stringify(result, null, 2))
log(`done: ${result.claimsPerTurn} claims/turn, supported ${precision.supported}, pass=${result.pass.met} kill=${result.kill.met}`)
