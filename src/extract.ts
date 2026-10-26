// Claim extraction from one conversation session. One model call per session
// (sessions are ~20-30 turns), cached on disk by (prompt version, model, input).
// The prompt is original to this repo; nothing is taken from HASH's
// proprietary AI worker.

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { CACHE, chat, parseJson, sha1 } from "./llm.ts"
import type { Claim } from "./ledger.ts"
import type { Turn } from "./locomo.ts"

export const EXTRACTOR_VERSION = process.env.OGM_EXTRACTOR ?? "claims@1"

const CITE_V2 = `
- PROVENANCE IS STRICT. "turns" must contain EVERY turn a reader needs to verify every word of the claim and its qualifiers: the turn that states it, plus any earlier turn that identifies who "them"/"it"/"that" is, what the event was, or when. If a detail cannot be verified from the turns you cite, drop the detail.
- Never add a detail that no turn states (no guessed locations, days, species, or reasons).`

const SYSTEM_V1 = `You turn a chat transcript into an exhaustive list of atomic, sourced claims for a long-term memory.

Rules:
- One claim = one subject, one predicate, one object. Split compound statements.
- "text" is a standalone sentence a stranger understands with no transcript: use full names, never pronouns ("she", "my", "we"), and start with the subject's name.
- Cover everything a friend might later ask about: events, plans, activities, possessions, pets, family, relationships, jobs, places, likes/dislikes, opinions, feelings, goals, things shown in shared images, and things said about other people.
- Put details that are not part of the triple (time, place, amount, duration, with-whom, reason) in "qualifiers".
- "when": the date the claimed thing happened or became true, resolved against the session date. "yesterday" in a session dated 2023-05-08 is "2023-05-07"; "last week" is "2023-05-01/2023-05-07"; "last year" is "2022"; "next month" is "2023-06". Use null if the transcript gives no time; do NOT default to the session date unless the turn says it is happening now or today.
- "turns": the ids (like D3:5) of every turn the claim is based on. A claim that needs two turns (a question and its answer) cites both.
- "confidence": 0-1, how directly the transcript states it (1 = stated outright, 0.5 = implied).
- Skip greetings and small talk that carry no information.

Return JSON: {"claims":[{"subject":"","predicate":"","object":"","qualifiers":{},"text":"","when":null,"turns":[""],"confidence":1}]}`

const SYSTEM = EXTRACTOR_VERSION === "claims@2" ? SYSTEM_V1.replace("\n\nReturn JSON", `${CITE_V2}\n\nReturn JSON`) : SYSTEM_V1

interface RawClaim { subject?: string; predicate?: string; object?: string | null; qualifiers?: Record<string, unknown>; text?: string; when?: string | null; turns?: string[]; confidence?: number }

function splitWhen(when: string | null | undefined): { validFrom: string | null; validTo: string | null } {
  if (!when || typeof when !== "string") return { validFrom: null, validTo: null }
  const [a, b] = when.split("/")
  return { validFrom: a?.trim() || null, validTo: b?.trim() || null }
}

export async function extractSession(model: string, conv: number, sessionKey: string, date: Date, rawDate: string, turns: Turn[]): Promise<Claim[]> {
  const transcript = turns.map((t) => `[${t.id}] ${t.text}`).join("\n")
  const user = `Session date: ${rawDate} (${date.toISOString().slice(0, 10)}).\n\nTranscript:\n${transcript}`
  const key = sha1(`${EXTRACTOR_VERSION}|${model}|${SYSTEM}|${user}`)
  const dir = join(CACHE, "extract")
  const path = join(dir, `${key}.json`)
  let raw: RawClaim[]
  if (existsSync(path)) raw = JSON.parse(readFileSync(path, "utf8")) as RawClaim[]
  else {
    const out = await chat(model, [{ role: "system", content: SYSTEM }, { role: "user", content: user }], { json: true, maxTokens: 16000 })
    raw = parseJson<{ claims?: RawClaim[] }>(out).claims ?? []
    mkdirSync(dir, { recursive: true })
    writeFileSync(path, JSON.stringify(raw))
  }
  const valid = new Set(turns.map((t) => t.id))
  const recordedAt = date.toISOString()
  return raw
    .filter((r) => r.text && r.subject)
    .map((r, i): Claim => ({
      id: `c${conv}-${sessionKey}-${i}`,
      kind: "assert",
      subject: String(r.subject).trim(),
      predicate: String(r.predicate ?? "").trim(),
      object: r.object == null ? null : String(r.object).trim(),
      qualifiers: Object.fromEntries(Object.entries(r.qualifiers ?? {}).map(([k, v]) => [k, String(v)])),
      text: String(r.text).trim(),
      ...splitWhen(r.when),
      recordedAt,
      sources: (r.turns ?? []).map(String).filter((t) => valid.has(t)),
      confidence: typeof r.confidence === "number" ? r.confidence : 1,
      actor: "ai",
      extractor: `${EXTRACTOR_VERSION}/${model}`,
    }))
    .filter((c) => c.sources.length > 0) // no source, no claim
}
