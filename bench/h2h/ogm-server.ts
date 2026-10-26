// E7 adapter: open-graph-memory behind the Mem0 OSS REST surface that Mem0's
// own benchmark runner (mem0ai/memory-benchmarks) speaks, so that runner can
// score us unmodified, side by side with Mem0, Graphiti and the others.
//
//   POST   /memories {messages, user_id, timestamp?} → {results:[{id, memory, event}]}
//   POST   /search   {query, user_id, limit?}         → {results:[{id, memory, score, created_at}]}
//   GET    /memories?user_id=…                         → {results:[…]}
//   DELETE /memories?user_id=…                         → {message}
//   GET    /health
//
// Adds only append turns. A session is every add with the same `timestamp`
// (the runner sends the session's date). Claims are extracted one session at
// a time, the extractor's normal unit, lazily on the first search after new
// adds; nothing about the questions is visible at extraction time.
//
// OGM_H2H_MODE picks the read path: dense | denseX | r2a (rrf of dense,
// denseX, claims and the facet leg; RESULTS.md "R2-A"). Results are chat
// messages; `score` is rank-derived ((n - i) / n) because the runner re-sorts
// by score and RRF has no meaningful per-hit score. `created_at` is the
// session date, which the runner prints as the memory's date.
//
//   OGM_EXTRACTOR=claims@2 OGM_H2H_MODE=r2a node bench/h2h/ogm-server.ts --port 8917

import { createServer, type IncomingMessage, type ServerResponse } from "node:http"
import { parseArgs } from "node:util"
import { extractSession, EXTRACTOR_VERSION } from "../../src/extract.ts"
import { planQuery, tagClaims } from "../../src/facets.ts"
import { project } from "../../src/graph.ts"
import type { Claim } from "../../src/ledger.ts"
import { embedMany, mapLimit } from "../../src/llm.ts"
import type { Turn } from "../../src/locomo.ts"
import { claimsRank, dense, denseX, facetRank, rrf, type Index, type Ranking } from "../../src/retrieve.ts"

const { values: a } = parseArgs({ options: { port: { type: "string", default: "8889" }, host: { type: "string", default: "127.0.0.1" } } })
const MODE = process.env.OGM_H2H_MODE ?? "r2a"
const MODEL = process.env.OGM_MODEL ?? "claude-haiku-5-5"
if (!["dense", "denseX", "r2a"].includes(MODE)) throw new Error(`OGM_H2H_MODE must be dense, denseX or r2a, not ${MODE}`)
if (!/^(127\.|::1$|localhost$)/.test(a.host!)) throw new Error(`no auth: refusing to bind ${a.host}; use a loopback address`)

interface Session { key: string; ts: number; turns: Turn[] }
interface Built { ix: Index; tags: Map<string, string[]> }
interface User { sessions: Map<number, Session>; speakers: Set<string>; built?: Promise<Built>; dirty: boolean }
const users = new Map<string, User>()
const log = (m: string) => console.log(`[ogm-h2h ${new Date().toISOString().slice(11, 19)}] ${m}`)

function user(id: string): User {
  let u = users.get(id)
  if (!u) users.set(id, (u = { sessions: new Map(), speakers: new Set(), dirty: true }))
  return u
}

function add(u: User, messages: Array<{ role?: string; content?: string }>, ts: number): Array<{ id: string; memory: string; event: string }> {
  let s = u.sessions.get(ts)
  if (!s) {
    s = { key: `session_${u.sessions.size + 1}`, ts, turns: [] }
    u.sessions.set(ts, s)
  }
  const sessionNo = [...u.sessions.values()].indexOf(s) + 1
  const out = []
  for (const m of messages) {
    const text = String(m.content ?? "").trim()
    if (!text) continue
    const speaker = /^([A-Z][\w'-]*(?: [A-Z][\w'-]*)?): /.exec(text)?.[1] ?? m.role ?? "user"
    u.speakers.add(speaker)
    const prevTurn = s.turns.at(-1)
    const t: Turn = { id: `D${sessionNo}:${s.turns.length + 1}`, session: s.key, sessionIndex: sessionNo - 1, date: new Date(ts * 1000), speaker, text, ...(prevTurn ? { prev: prevTurn.text } : {}) }
    s.turns.push(t)
    out.push({ id: t.id, memory: text, event: "ADD" })
  }
  u.dirty = true
  return out
}

async function build(u: User, uid: string): Promise<Built> {
  const sessions = [...u.sessions.values()].sort((x, y) => x.ts - y.ts)
  const turns = sessions.flatMap((s) => s.turns)
  const t0 = Date.now()
  let claims: Claim[] = []
  if (MODE !== "dense") {
    const per = await mapLimit(sessions, 4, (s) => {
      const d = new Date(s.ts * 1000)
      return extractSession(MODEL, 0, `${uid}-${s.key}`, d, d.toUTCString(), s.turns)
    })
    claims = per.flat()
  }
  const byTurn = new Map<string, string[]>()
  for (const c of claims) for (const s of c.sources) byTurn.set(s, [...(byTurn.get(s) ?? []), c.text])
  const ix: Index = {
    turns,
    turnVec: await embedMany(turns.map((t) => (t.prev ? `${t.prev}\n${t.text}` : t.text))),
    turnVecX: MODE === "dense" ? undefined : await embedMany(turns.map((t) => `${t.prev ? `${t.prev}\n` : ""}${t.text}${byTurn.has(t.id) ? `\nFacts: ${byTurn.get(t.id)!.join(" ")}` : ""}`)),
    claims,
    claimVec: await embedMany(claims.map((c) => `${c.text}${c.validFrom ? ` (${c.validFrom}${c.validTo ? ` to ${c.validTo}` : ""})` : ""}`)),
    graph: await project(claims, turns, { synonymThreshold: 1, turnAdjacency: 0 }),
    pos: new Map(turns.map((t, i) => [t.id, i])),
  }
  const tags = MODE === "r2a" ? await tagClaims(MODEL, claims) : new Map<string, string[]>()
  log(`built ${uid}: ${sessions.length} sessions, ${turns.length} turns, ${claims.length} claims (${EXTRACTOR_VERSION}), ${((Date.now() - t0) / 1000).toFixed(0)} s`)
  return { ix, tags }
}

async function search(uid: string, query: string, limit: number) {
  const u = users.get(uid)
  if (!u || !u.sessions.size) return []
  if (u.dirty || !u.built) {
    u.dirty = false
    u.built = build(u, uid)
    u.built.catch(() => { u.dirty = true })
  }
  const { ix, tags } = await u.built
  const [q] = await embedMany([query])
  const base = dense(ix, q!)
  let r: Ranking
  if (MODE === "dense") r = base
  else if (MODE === "denseX") r = denseX(ix, q!)
  else {
    const plan = await planQuery(MODEL, query, [...u.speakers])
    r = rrf([base, denseX(ix, q!), claimsRank(ix, q!, base), facetRank(ix, q!, plan, tags, base)])
  }
  const top = r.slice(0, limit)
  const n = top.length
  return top.map((id, i) => {
    const t = ix.turns[ix.pos.get(id)!]!
    const when = t.date.toISOString()
    return { id, memory: t.text, score: (n - i) / n, created_at: when, updated_at: when, metadata: { ogm_mode: MODE } }
  })
}

async function body(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = []
  for await (const c of req) chunks.push(c as Buffer)
  return chunks.length ? (JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>) : {}
}

function send(res: ServerResponse, status: number, data: unknown): void {
  res.writeHead(status, { "content-type": "application/json" })
  res.end(JSON.stringify(data))
}

createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? "/", "http://x")
    if (req.method === "GET" && url.pathname === "/health") return send(res, 200, { status: "ok", backend: "open-graph-memory", mode: MODE, extractor: EXTRACTOR_VERSION })
    if (req.method === "POST" && url.pathname === "/memories") {
      const b = await body(req)
      const ts = typeof b.timestamp === "number" ? b.timestamp : Math.floor(Date.now() / 1000)
      return send(res, 200, { results: add(user(String(b.user_id)), (b.messages as Array<{ role?: string; content?: string }>) ?? [], ts) })
    }
    if (req.method === "POST" && url.pathname === "/search") {
      const b = await body(req)
      return send(res, 200, { results: await search(String(b.user_id), String(b.query ?? ""), Math.min(Number(b.limit ?? 100), 1000)) })
    }
    const uid = url.searchParams.get("user_id") ?? ""
    if (req.method === "GET" && url.pathname === "/memories") {
      const u = users.get(uid)
      return send(res, 200, { results: u ? [...u.sessions.values()].flatMap((s) => s.turns.map((t) => ({ id: t.id, memory: t.text, created_at: t.date.toISOString() }))) : [] })
    }
    if (req.method === "DELETE" && url.pathname === "/memories") {
      users.delete(uid)
      return send(res, 200, { message: "deleted" })
    }
    send(res, 404, { error: "not found" })
  } catch (e) {
    log(`error: ${(e as Error).message}`)
    send(res, 500, { error: (e as Error).message.slice(0, 300) })
  }
}).listen(Number(a.port), a.host, () => log(`on http://${a.host}:${a.port} (mode ${MODE}, extractor ${EXTRACTOR_VERSION}, model ${MODEL}; no auth, loopback only)`))
